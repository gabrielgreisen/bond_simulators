import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import QuantLib as ql
import time
from scipy.stats import qmc
from convertible_pricer_class import Tsiveriotis_Fernandes_Pricer

def simulation(N,
               chunk_size=15_000,
               out_dir="simulation_output", worker_id=0
               ):
      # Maintain number of simulations within bounds
      if N < 100 or N > 10_000_000:
            raise ValueError("N must be between 100 and 10,000,000")

      calendar = ql.UnitedStates(ql.UnitedStates.GovernmentBond)
      todays_date = calendar.adjust(ql.Date.todaysDate())
      ql.Settings.instance().evaluationDate = todays_date
      day_count = ql.Actual365Fixed()

      pricer = Tsiveriotis_Fernandes_Pricer(
            todays_date=todays_date,
            calendar=calendar,
            day_count=day_count,
            steps_binomial=20_000,
      )

      frequencies = [ql.Annual, ql.Semiannual, ql.Quarterly]

      # Edge-case cells: each defines [S, r, q, bs_vol, credit_spread, coupon_rate, maturity_years, conversion_ratio]
      cells = [
            # Cell 1 — Deep ITM × Short Maturity
            (np.array([200, 0.01, 0.00, 0.10, 0.005, 0.00, 0.25, 5.0]),
             np.array([500, 0.09, 0.06, 0.80, 0.15,  0.10, 2.5,  10.0])),
            # Cell 2 — Deep ITM × Long Maturity
            (np.array([200, 0.01, 0.00, 0.10, 0.005, 0.00, 7.5, 5.0]),
             np.array([500, 0.09, 0.06, 0.80, 0.15,  0.10, 10.0, 10.0])),
            # Cell 3 — Deep OTM × Short Maturity
            (np.array([10,  0.01, 0.00, 0.10, 0.005, 0.00, 0.25, 1.0]),
             np.array([40,  0.09, 0.06, 0.80, 0.15,  0.10, 2.5,  3.0])),
            # Cell 4 — Deep OTM × Long Maturity
            (np.array([10,  0.01, 0.00, 0.10, 0.005, 0.00, 7.5, 1.0]),
             np.array([40,  0.09, 0.06, 0.80, 0.15,  0.10, 10.0, 3.0])),
            # Cell 5 — Full Moneyness × Sub-1y Maturity
            (np.array([10,  0.01, 0.00, 0.10, 0.005, 0.00, 0.25, 1.0]),
             np.array([500, 0.09, 0.06, 0.80, 0.15,  0.10, 1.0,  10.0])),
      ]

      # Split N equally across cells, remainder goes to last cell
      n_cells = len(cells)
      base_n = N // n_cells
      cell_sizes = [base_n] * n_cells
      cell_sizes[-1] += N - base_n * n_cells

      # Latin Hypercube Sampling per cell, concatenated
      sampler = qmc.LatinHypercube(d=8)
      lhs_blocks = []
      for (lo, hi), n_i in zip(cells, cell_sizes):
            unit = sampler.random(n=n_i)
            lhs_blocks.append(qmc.scale(unit, lo, hi))
      lhs_scaled = np.vstack(lhs_blocks)

      # Shuffle so cells are interleaved across chunks
      rng = np.random.default_rng()
      rng.shuffle(lhs_scaled)

      data = []
      chunk = 0
      start_time = time.perf_counter()

      for i in range(N):

            if (not(i % chunk_size) and (not(i == 0))):
                  pd.DataFrame(data).to_csv(f"{out_dir}/w{worker_id}_simulation_chunk{chunk}.csv", index=False)
                  chunk += 1
                  print(f"Worker {worker_id} chunk {chunk-1} time {time.perf_counter() - start_time:.1f}s")
                  start_time = time.perf_counter()
                  data.clear()

            # Market parameters
            S             = lhs_scaled[i, 0]
            r             = lhs_scaled[i, 1]
            q             = lhs_scaled[i, 2]
            bs_vol        = lhs_scaled[i, 3]
            credit_spread = lhs_scaled[i, 4]

            # Bond structure
            redemption = 1000.0

            coupon_rate = lhs_scaled[i, 5]
            settlement_days = 2
            frequency = frequencies[rng.integers(0, len(frequencies))]

            # Maturity: 1–10 years from today
            maturity_years = lhs_scaled[i, 6]
            maturity_days = int(round(maturity_years * 365))
            issue_date = todays_date
            maturity_date = calendar.advance(todays_date, ql.Period(maturity_days, ql.Days))

            # Conversion
            conversion_ratio = lhs_scaled[i, 7]
            conversion_price = redemption / conversion_ratio

            # Price; protect the whole job from a single bad draw
            try:
                  price = pricer.price_vanilla(
                        redemption=redemption,
                        spot_price=S,
                        conversion_ratio=conversion_ratio,
                        issue_date=issue_date,
                        maturity_date=maturity_date,
                        coupon_rate=coupon_rate,
                        frequency=frequency,
                        settlement_days=settlement_days,
                        r=r,
                        q=q,
                        bs_volatility=bs_vol,
                        credit_spread_rate=credit_spread,
                  )
            except Exception:
                  price = np.nan

            data.append({
                  "S": S, "r": r, "q": q,
                  "bs_vol": bs_vol, "credit_spread": credit_spread,
                  "redemption": redemption, 
                  "coupon_rate": coupon_rate, "frequency": frequency,
                  "maturity_years": maturity_years,
                  "conversion_ratio": conversion_ratio,
                  "conversion_price": conversion_price,
                  "price_convertible": price,
            })

      pd.DataFrame(data).to_csv(f"{out_dir}/w{worker_id}_simulation_chunk{chunk}.csv", index=False)
      print(f"Worker {worker_id} chunk {chunk} time {time.perf_counter() - start_time:.1f}s (final)")
