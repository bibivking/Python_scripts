##/usr/bin/env python
"""
AWAP weather generator functions
- testing before writing them in C for GDAY
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from math import pi, cos, sin, exp, sqrt, acos, asin, tan
import random

class WeatherGenerator(object):

    def __init__(self, lat, lon):

        # conversion from SW (MJ m-2 d-1) to PAR (MJ m-2 d-1)
        self.SW_2_PAR_MJ = 0.5

        # conversion from SW (W m-2) to PAR (umol m-s s-1)
        self.SW_2_PAR = 2.3
        self.PAR_2_SW = 1.0 / self.SW_2_PAR
        self.J_TO_MJ = 1E-6
        self.SEC_2_DAY = 86400.0
        self.MJ_TO_J = 1E6
        self.DAY_2_SEC = 1.0 / self.SEC_2_DAY
        self.SEC_2_HLFHR = 1800.0
        self.HLFHR_2_SEC = 1.0 / self.SEC_2_HLFHR
        self.J_TO_UMOL = 4.57
        self.UMOL_TO_J = 1.0 / self.J_TO_UMOL
        self.UMOLPERJ = 4.57     # Conversion from J to umol quanta
        self.PiBy2    = 1.57079632
        self.lat = lat
        self.lon = lon

    def maestra_diurnal_func(self, doy, tmin, tmax):
        """ Not sure where this function original comes from... """
        tav = (tmax + tmin) / 2.0
        tampl = (tmax - tmin) / 2.0

        dayl = self.calc_day_length(doy, 365) * 2.

        tday = np.zeros(48)
        for i in range(1, 48+1):
            hrtime = i - 0.5
            time = i + dayl * 0.5 - 48.0 / 2.0
            if time < 0.0 or time > dayl:
                if time < 0.0:
                    hrtime += 48

                arg1 = tav
                arg2 = (tav - tmin) * (hrtime - dayl * 0.5 - (48.0 / 2.0))
                arg3 = 48.0 - dayl

                tday[i-1] = arg1 - arg2 / arg3
            else:
                tday[i-1] = tav - tampl * cos(1.5 * pi * time / dayl)

        return (tday)

    def estimate_diurnal_temp(self, doy, tmin, tmax):
        """
        Calculate diurnal temperature following Parton and Logan
        the day is divided into two segments and using a truncated sine wave
        in the daylight and an exponential decrease in temperature
        at night.
        TO DO:
        - Hours between 00:00 and sunrise should be modelled using the previous
        days information.
        Arguments:
        ----------
        tmin : double
             day minimum temp (deg C)
        tmax : float
             day maximum temp (deg C)
        References:
        ----------
        * Parton and Logan (1981) A model for dirunal variation in soil and
          air temperature. Agricultural Meteorology, 23, 205--216.
        * Kimball and Bellamy (1986) Energy in Agriculture, 5, 185-197.
        """
        # 1.5 m air temperature values from Parton and Logan, table 1
        a = 1.86
        b = 2.2     # nighttime coeffcient
        c = -0.17   # lag of the min temp from the time of runrise

        day_length = self.calc_day_length(doy, 365)

        night_length = 24 - day_length

        sunrise = 12.0 - day_length / 2.0 + c
        sunset = 12.0 + day_length / 2.0

        # temperature at sunset
        m = sunset - sunrise + c
        tset = (tmax - tmin) * sin(pi * m / (day_length + 2.0 * a)) + tmin

        tday = np.zeros(48)
        for i in range(1, 48+1):
            hour = i / 2.0

            # hour - time of the minimum temperature (accounting for lag time)
            m = hour - sunrise + c
            if hour >= sunrise and hour <= sunset:
                tday[i-1] = tmin + (tmax - tmin) * \
                          sin((pi * m) / (day_length + 2.0 * a))
            else:
                if hour > sunset:
                    n = hour - sunset
                elif hour < sunrise:
                    n = (24.0 + hour) - sunset

                d = (tset - tmin) / (exp(b) - 1.0)

                # includes missing displacement to allow T to reach Tmin, this
                # removes a discontinuity in the original Parton and Logan eqn.
                # See Kimball and Bellamy (1986) Energy in Agriculture, 5,
                # 185-197
                tday[i-1] = (tmin -d) + (tset - tmin - d) * \
                          exp(-b * n / (night_length + c))

        return (tday)

    def estimate_diurnal_temp_cable(self, doy, tmin, tmax, tminnext, tmaxprev):
        # -----------
        # Temperature
        # -----------
        # These are parameters required for the calculation of temperature according to
        # Cesaraccio et al 2001 for sunrise-to-sunrise-next-day. Because we are calculating temp for
        # midnight-to-midnight, we need to calculate midnight-to-sunrise temps using data for the
        # previous day (-24h), hence the extra parameters denoted by *'s below which are not
        # mentioned per se in Cesaraccio. Cesaraccio symbology for incoming met data is included
        # here as comments for completeness:
        #    Sym in Cesaraccio et al 2001
        # TempMinDay                                                        Tn
        # TempMaxDay                                                        Tx
        # TempMinDayNext                                                    Tp
        # TempMaxDayPrev

        LatRad  = self.lat*pi/180.0                 # latitude in radians
        YearRad = 2.0*pi*(doy-1)/365.0              # day of year in radians
        DecRad = 0.006918 - 0.399912*cos(YearRad) + 0.070257*sin(YearRad)   \
               - 0.006758*cos(2.0*YearRad) + 0.000907*sin(2.0*YearRad)      \
               - 0.002697*cos(3.0*YearRad) + 0.001480*sin(3.0*YearRad)
        TanTan = -tan(LatRad)*tan(DecRad)
        HDLRad = acos(TanTan)
        DayLength = 24.0*2.0*HDLRad / (2.0*pi)

        TimeSunrise = (acos(min(tan(LatRad)*tan(DecRad),0.9999)))*12./pi  # Hn
        TimeSunset  = TimeSunrise + DayLength                             # Ho
        TimeMaxTemp = TimeSunset - min(4.,( DayLength * 0.4))             # Hx

        TimeSunsetPrev    = TimeSunset - 24.     # * Ho-24h (a negative hour)
        TimeSunriseNext   = TimeSunrise + 24.    # Hp
        TempSunset        = tmax - (0.39 * (tmax - tminnext))  # To
        TempSunsetPrev    = tmaxprev - (0.39 * (tmaxprev - tmin))  # * To-24h
        TempRangeDay      = tmax - tmin                     # alpha = Tx-Tn
        TempRangeAft      = tmax - TempSunset               # R = Tx-To
        TempNightRate     = (tminnext - TempSunset)/ \
                            sqrt(TimeSunriseNext-TimeSunset)   # b = (Tp-To)/sqrt(Hp-Ho)
        TempNightRatePrev = (tmin - TempSunsetPrev)/ \
                            sqrt(TimeSunrise-TimeSunsetPrev)
        # * b-24h = (Tn-(To-24h))/sqrt(Hn-(Ho-24h))

        # -----------
        # Temperature # Tair/GSWP3.BC.Tair.3hrMap
        # -----------
        # Calculate temperature according to Cesaraccio et al 2001, including midnight to
        # sunrise period using previous days info, and ignoring the period from after the
        # following midnight to sunrise the next day, normally calculated by Cesaraccio.

        tday = np.zeros(48)
        for i in range(1, 48+1):
            hour = i / 2.0
            if hour <= TimeSunrise :  # Midnight to sunrise
                tday[i-1] = TempSunsetPrev + TempNightRatePrev * \
                            sqrt(hour - TimeSunsetPrev)
            elif hour > TimeSunrise and hour <= TimeMaxTemp: # Sunrise to time of maximum temperature
                tday[i-1] = tmin + TempRangeDay * sin(((hour - TimeSunrise)/ \
                            (TimeMaxTemp-TimeSunrise))*self.PiBy2 )
            elif hour > TimeMaxTemp and hour <= TimeSunset: # Time of maximum temperature to sunset
                tday[i-1] = TempSunset + TempRangeAft * \
                            sin(self.PiBy2  + ((hour-TimeMaxTemp)/4.*self.PiBy2 ))
            elif hour > TimeSunset: # Sunset to midnight
                tday[i-1] = TempSunset + TempNightRate * sqrt(hour - TimeSunset)
        return (tday)

    def calc_day_length(self, doy, yr_days):

        """
        Daylength in hours
        Eqns come from Leuning A4, A5 and A6, pg. 1196
        Reference:
        ----------
        Leuning et al (1995) Plant, Cell and Environment, 18, 1183-1200.
        Parameters:
        -----------
        doy : int
            day of year, 1=jan 1
        yr_days : int
            number of days in a year, 365 or 366
        latitude : float
            latitude [degrees]
        Returns:
        --------
        dayl : float
            daylength [hrs]
        """

        deg2rad = pi / 180.0;
        rlat = self.lat * deg2rad;
        sindec = -sin(23.5 * deg2rad) * cos(2.0 * pi * (doy + 10.0) / yr_days);
        a = sin(rlat) * sindec;
        b = cos(rlat) * cos(asin(sindec));

        return 12.0 * (1.0 + (2.0 / pi) * asin(a / b))

if __name__ == "__main__":

    lat = -23.575001
    lon = 152.524994
    doy = 180.0
    tmin = 2.0
    tmax = 24.0
    doy = 180.0
    lat = 50.0
    tminnext = 2.0
    tmaxprev = 24.0

    hours = np.arange(48) / 2.0

    WG = WeatherGenerator(lat, lon)

    tday  = WG.estimate_diurnal_temp(doy, tmin, tmax)
    tday2 = WG.maestra_diurnal_func(doy, tmin, tmax)
    tday3 = WG.estimate_diurnal_temp_cable(doy, tmin, tmax, tminnext, tmaxprev)

    plt.plot(hours, tday, "r-", label="Parton & Logan")
    plt.plot(hours, tday2, "k-", label="MAESPA")
    plt.plot(hours, tday3, "c-", label="Cesaraccio")
    plt.legend(numpoints=1, loc="best")
    plt.ylabel("Air Temperature (deg C)")
    plt.xlabel("Hour of day")
    plt.show()
