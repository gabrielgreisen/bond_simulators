import QuantLib as ql
import numpy as np

class Tsiveriotis_Fernandes_Pricer:
    """
    Tsiveriotis-Fernandes convertible bond pricer.

    Uses QuantLib's BinomialConvertibleEngine (CRR tree) which
    internally splits the convertible value into equity and debt
    components, discounting each at the appropriate rate.

    Reuses (built once in __init__, updated via SimpleQuote.setValue):
      - spot, r, q, volatility, credit_spread quote handles
      - flat term structures (observe the quote handles)
      - BlackScholesMertonProcess (observes the term structures)

    Rebuilds per-call (structural params differ across companies):
      - callability schedule, coupon schedule, exercise
      - ConvertibleFixedCouponBond instrument
      - BinomialConvertibleEngine (cheap; tree built during NPV())
    """

    def __init__(
        self,
        todays_date: ql.Date,
        calendar: ql.Calendar,
        day_count: ql.DayCounter,
        steps_binomial=20_000,
        spot_init=100.0,
        r_init=0.04,
        q_init=0.02,
        vol_init=0.30,
        credit_spread_init=0.03,
    ):
        self.todays_date = todays_date
        self.calendar = calendar
        self.day_count = day_count
        self.steps_binomial = int(steps_binomial)

        # Anchor QuantLib to valuation date
        ql.Settings.instance().evaluationDate = self.todays_date

        # --- Mutable quotes (updated cheaply via .setValue per row) ---
        self._spot_q = ql.SimpleQuote(float(spot_init))
        self._r_q = ql.SimpleQuote(float(r_init))
        self._q_q = ql.SimpleQuote(float(q_init))
        self._vol_q = ql.SimpleQuote(float(vol_init))
        self._cs_q = ql.SimpleQuote(float(credit_spread_init))

        # --- Term structures driven by quote handles ---
        spot_handle = ql.QuoteHandle(self._spot_q)
        r_handle = ql.QuoteHandle(self._r_q)
        q_handle = ql.QuoteHandle(self._q_q)
        vol_handle = ql.QuoteHandle(self._vol_q)

        self._rf_curve = ql.YieldTermStructureHandle(
            ql.FlatForward(self.todays_date, r_handle, self.day_count)
        )
        self._div_curve = ql.YieldTermStructureHandle(
            ql.FlatForward(self.todays_date, q_handle, self.day_count)
        )
        self._vol_surface = ql.BlackVolTermStructureHandle(
            ql.BlackConstantVol(
                self.todays_date, self.calendar, vol_handle, self.day_count,
            )
        )

        # --- BSM process (observes all handles above) ---
        self._bsm_process = ql.BlackScholesMertonProcess(
            spot_handle, self._div_curve, self._rf_curve, self._vol_surface,
        )

        # --- Credit spread handle ---
        self._cs_handle = ql.QuoteHandle(self._cs_q)

        # --- Empty dividend schedule (continuous yield via BSM process) ---
        self._div_schedule = ql.DividendSchedule()

    def set_market(
        self,
        spot: float,
        r: float,
        q: float,
        bs_volatility: float,
        credit_spread_rate: float,
    ):
        """Update all market quotes in place (cheap, no object rebuilds)."""
        self._spot_q.setValue(float(spot))
        self._r_q.setValue(float(r))
        self._q_q.setValue(float(q))
        self._vol_q.setValue(float(bs_volatility))
        self._cs_q.setValue(float(credit_spread_rate))

    def price_vanilla(
        self,
        redemption,        # Par value at maturity
        spot_price,        # Current stock price
        conversion_ratio,  # Shares per bond upon conversion (QL uses $100 face, scale accordingly)
        issue_date,
        maturity_date,
        coupon_rate,       # Annual coupon rate (e.g. 0.0575)
        frequency,         # Coupon frequency (e.g. ql.Semiannual)
        settlement_days,   # Settlement days (e.g. 2)
        r,                 # Risk-free rate
        q,                 # Continuous dividend yield
        bs_volatility,     # Black-Scholes flat volatility
        credit_spread_rate,
        call_dates=None,   # List of ql.Date for issuer calls
        call_price=None,   # Clean call price
        put_dates=None,    # List of ql.Date for holder puts
        put_price=None,    # Clean put price
    ) -> float:

        # Update market quotes (propagates through handles → curves → process)
        self.set_market(spot_price, r, q, bs_volatility, credit_spread_rate)

        # --- Callability schedule (rebuilt: differs per bond) ---
        callability_schedule = ql.CallabilitySchedule()

        if call_dates and call_price is not None:
            for call_date in call_dates:
                bond_call_price = ql.BondPrice(float(call_price), ql.BondPrice.Clean)
                callability_schedule.append(
                    ql.Callability(bond_call_price, ql.Callability.Call, call_date)
                )

        if put_dates and put_price is not None:
            for put_date in put_dates:
                bond_put_price = ql.BondPrice(float(put_price), ql.BondPrice.Clean)
                callability_schedule.append(
                    ql.Callability(bond_put_price, ql.Callability.Put, put_date)
                )

        # --- Coupon schedule (rebuilt: differs per bond) ---
        tenor = ql.Period(frequency)
        schedule = ql.Schedule(
            issue_date, maturity_date, tenor,
            self.calendar, ql.Unadjusted, ql.Unadjusted,
            ql.DateGeneration.Backward, False,
        )

        # --- Exercise (rebuilt: maturity differs per bond) ---
        exercise = ql.AmericanExercise(self.todays_date, maturity_date)

        # --- Convertible bond instrument (rebuilt per call) ---
        bond = ql.ConvertibleFixedCouponBond(
            exercise,
            float(conversion_ratio),
            callability_schedule,
            issue_date,
            int(settlement_days),
            [float(coupon_rate)],
            self.day_count,
            schedule,
            float(redemption),
        )

        # --- Engine (rebuilt: cheap, tree built during NPV()) ---
        engine = ql.BinomialConvertibleEngine(
            self._bsm_process, "crr", self.steps_binomial,
            self._cs_handle, self._div_schedule,
        )

        bond.setPricingEngine(engine)

        try:
            return float(bond.NPV())
        except RuntimeError:
            return np.nan
