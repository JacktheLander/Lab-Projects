import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- Load LMP data ---
lmp_path = "LMP_6.6.25.csv"
lmp_df = pd.read_csv(lmp_path)
lmp_df = lmp_df[(lmp_df['LMP_TYPE'] == 'LMP') & (lmp_df['NODE'] == '0096WD_7_N001')].copy()
lmp_df['INTERVALSTARTTIME_GMT'] = pd.to_datetime(lmp_df['INTERVALSTARTTIME_GMT'])
lmp_df = lmp_df.sort_values('INTERVALSTARTTIME_GMT')
lmp_df = lmp_df.set_index('INTERVALSTARTTIME_GMT')
lmp_15min = lmp_df['MW'].resample('15min').ffill()

# --- Load Solar Data and Resample ---
solar_df = pd.read_csv("SolarOutput_6.6.06.csv")
solar_df['LocalTime'] = pd.to_datetime(solar_df['LocalTime'], format='%m/%d/%y %H:%M')
solar_df.set_index('LocalTime', inplace=True)
solar_day = solar_df.loc['2006-06-06']
solar_15min = solar_day['Power(MW)'].resample('15min').mean()

for N_packs in range(5, 101, 5):
    # --- Megapack2xl Properties ---
    E_per_pack = 3.916  # MWh
    P_per_pack = 1.927  # MW
    efficiency = 0.92
    Δt_hr = 0.25  # 15 minutes = 0.25 hours

    E_max = N_packs * E_per_pack
    P_max = N_packs * P_per_pack


    # --- Align both time series ---
    timesteps = min(len(lmp_15min), len(solar_15min))
    prices = lmp_15min[:timesteps].values
    solar_output = solar_15min[:timesteps].values
    times = lmp_15min.index[:timesteps]

    # --- Initialize ---
    charge_power = np.zeros(timesteps)
    discharge_power = np.zeros(timesteps)
    soc = np.zeros(timesteps + 1)

    solar_direct_sell = np.zeros(timesteps)  # MW sold directly to grid

    # --- Simulation Loop ---
    for t in range(timesteps):
        hour = times[t].hour
        price = prices[t]
        solar = solar_output[t]
        soc_now = soc[t]

        # --- Charging Window: 15:00 onwards UTC ---
        if 15 <= hour <= 23 and soc_now < E_max:
            charge = min(P_max, solar, (E_max - soc_now) / (efficiency * Δt_hr))
            charge_power[t] = charge
            soc[t + 1] = soc_now + charge * efficiency * Δt_hr
            solar_direct_sell[t] = max(0, solar - charge)

        # --- Discharging Window: 02:00–03:59 UTC ---
        elif 2 <= hour <= 3 and soc_now > 0:
            discharge = min(P_max, soc_now * efficiency / Δt_hr)
            discharge_power[t] = discharge
            soc[t + 1] = soc_now - discharge / efficiency * Δt_hr
            solar_direct_sell[t] = solar  # solar not used for charging now

        # --- Not Charging or Discharging ---
        else:
            soc[t + 1] = soc_now
            solar_direct_sell[t] = solar  # sell all solar directly

    # --- Revenue and Profit ---
    revenue = (
        discharge_power * prices            # profit from selling stored energy
        + solar_direct_sell * prices        # profit from direct solar sales
    ) * Δt_hr

    cumulative_profit = np.cumsum(revenue)

    # --- Plot ---
    plt.figure(figsize=(15, 10))

    plt.subplot(3, 1, 1)
    plt.plot(times, prices, label='LMP ($/MWh)', color='purple')
    plt.title(f"Real LMP & Solar + {N_packs} Megapack 2XL EMS")
    plt.ylabel("Price ($/MWh)")
    plt.grid(True)

    plt.subplot(3, 1, 2)
    plt.plot(times, solar_output, label='Solar Output (MW)', color='orange')
    plt.plot(times, charge_power, label='Charge (MW)', color='green')
    plt.plot(times, discharge_power, label='Discharge (MW)', color='red')
    plt.ylabel("Power (MW)")
    plt.legend()
    plt.grid(True)

    plt.subplot(3, 1, 3)
    plt.plot(times, 50*soc[1:], label='State of Charge (MWh)', color='blue')
    plt.plot(times, cumulative_profit, label='Cumulative Profit ($)', color='black')
    plt.xlabel("Time (UTC)")
    plt.ylabel("Energy / Profit")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(f"Outputs/Predictive/{N_packs}packs_predictive.png")
    #plt.show()

    print(f"Total Arbitrage Profit for {N_packs} Megapacks: ${cumulative_profit[-1]:,.2f}")