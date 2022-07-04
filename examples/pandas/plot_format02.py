"""
Agg Patient Single Row
======================

Example
"""

# Generic
import warnings
import numpy as np
import pandas as pd 

# Ignore warnings
warnings.simplefilter("ignore")

def load_data():
	return pd.read_csv('./laboratory.csv', parse_dates=['date'])

def create_data():
	""""""
	# Configuration
	ROWS, COLS = 150000, 10
	PATIENTS = 300

	# Create random values
	features = np.random.random_sample((ROWS, COLS))
	patients = np.random.randint(PATIENTS, size=(ROWS, 1))

	# Create DataFrame
	df = pd.DataFrame(data=features)
	df = df.add_prefix('f_')
	df['id'] = patients

	# Return
	return df


# -----------------------
# Read data
# -----------------------
# Read data
#data = pd.read_csv('./laboratory.csv', parse_dates=['date'])

data = create_data()


# -----------------------
# Format
# -----------------------
# Configuration
show_progress_every = 10 # Number of patients
break_loop_after = 100  # Number of patients or None

# Create empty outcome
results = pd.DataFrame()

# Groups
groups = data.groupby(by='id')

# Step by step (16270 groups!)
for i, (k, g) in enumerate(groups):
	# Show information
	if (i % show_progress_every) == 0:
		print("%5s/%5s. Patient '%s'" % (i+1, len(groups), k))

	# Show dataframe
	#print(g)

	# Sort by dates (if needed)

	# Fill empty values
	#g.fillna(method='backfill', inplace=True)

	# Compute statistics
	# ------------------
	# .. note: Forward/backward filling does not affect
	#          the max/min but it affects the mean or
	#          median.
	#
	# .. note: You could also create a map with all the 
	#          functions you want to apply instead of 
	#          using describe. This is useful if you need 
	#          specific stats

	# Get the common stats
	#d = g.describe()

	# Get specific stats for all columns	
	d = g.agg({c: ['max', 'min'] for c in g.columns})

	# Stack the describe outcome 
	d = d.stack()
	d.index = ['_'.join(e) for e in d.index.tolist()]
	d['id'] = k                  # patient identifier
	#d['date'] = min(g['date'])   # admission date

	# Append result
	results = results.append(d, ignore_index=True)

	# Break clause for testing
	#if break_loop_after is not None:
	if i==break_loop_after:
		break


# Show columns
#print(results.columns.values)

print(results)

# Show results (partially)
#print(results[['id', 'date', 'max_wcc']])

# .. note: Once it works as you want, you can try to do it
#          in one single line and compare the results to 
#          verify that it is correct.