# Given data
current_age = 27  # current age of the individual
desired_age = 100  # age until which the individual wants the money to last
annual_expenditure = 600000  # annual expenditure in currency units
annual_inflation_rate = 0.03  # 8% annual inflation rate
annual_interest_rate = 0.035  # 3% annual interest rate

# Number of years the money needs to last
years = desired_age - current_age

# Function to calculate the present value of an annuity
def present_value_annuity(annual_payment, interest_rate, periods):
    pv = annual_payment * ((1 - (1 + interest_rate) ** -periods) / interest_rate)
    return pv

# Adjusting the annual expenditure for inflation each year
# and calculating the present value of these adjusted expenditures
total_pv = 0
for year in range(years):
    adjusted_annual_expenditure = annual_expenditure * ((1 + annual_inflation_rate) ** year)
    pv = present_value_annuity(adjusted_annual_expenditure, annual_interest_rate, 1)
    total_pv += pv

print(total_pv)
