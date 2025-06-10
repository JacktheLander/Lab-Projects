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


for N_packs in range(5, 101, 5):
    # --- Megapack2xl Properties ---
    E_per_pack = 3.916  # MWh
    P_per_pack = 1.927  # MW
    efficiency = 0.92
    Δt_hr = 0.25  # 15 minutes = 0.25 hours

    E_max = N_packs * E_per_pack
    P_max = N_packs * P_per_pack


    # --- Align both time series ---
    timesteps = len(lmp_15min)
    prices = lmp_15min[:timesteps].values
    times = lmp_15min.index[:timesteps]

    # --- Initialize ---
    charge_power = np.zeros(timesteps)
    discharge_power = np.zeros(timesteps)
    soc = np.zeros(timesteps + 1)

    # --- DCT - Demand Charge Threshold ---
    low_price = np.percentile(prices, 30)
    high_price = np.percentile(prices, 70)

    solar_direct_sell = np.zeros(timesteps)  # MW sold directly to grid

    # --- Simulation Loop ---
    for t in range(timesteps):
        price = prices[t]
        soc_now = soc[t]
        hour = times[t].hour

        # --- Charging Window: 15:00 until full UTC ---
        if 16 <= hour <= 23 and soc_now < E_max:
            charge = min(P_max, (E_max - soc_now) / (efficiency * Δt_hr))
            charge_power[t] = charge
            soc[t + 1] = soc_now + charge * efficiency * Δt_hr

        # --- Discharging Window: 02:00–03:59 UTC ---
        elif 2 <= hour <= 3 and soc_now > 0:
            discharge = min(P_max, soc_now * efficiency / Δt_hr)
            discharge_power[t] = discharge
            soc[t + 1] = soc_now - discharge / efficiency * Δt_hr

        # --- Hold SoC otherwise ---
        else:
            soc[t + 1] = soc_now

    # Calculate cost to charge and revenue from discharge
    cost_to_charge = charge_power * prices * Δt_hr
    revenue_from_discharge = discharge_power * prices * Δt_hr

    # Net arbitrage revenue
    revenue = revenue_from_discharge - cost_to_charge
    cumulative_profit = np.cumsum(revenue)

    # --- Plot ---
    plt.figure(figsize=(15, 10))

    plt.subplot(3, 1, 1)
    plt.plot(times, prices, label='LMP ($/MWh)', color='purple')
    plt.title(f"Real LMP & Solar + {N_packs} Megapack 2XL EMS")
    plt.ylabel("Price ($/MWh)")
    plt.grid(True)

    plt.subplot(3, 1, 2)
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
    plt.savefig(f"Outputs/Grid/{N_packs}packs_grid.png")
    #plt.show()

    print(f"Total Arbitrage Profit for {N_packs} Megapacks: ${cumulative_profit[-1]:,.2f}")