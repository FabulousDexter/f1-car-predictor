import fastf1

# Get session of GP Las Vegas 2025
session = fastf1.get_session(2025, "Las Vegas", "R")
session.load()

# Print columns of the session data
print(session.results.columns)

print(session.results["TeamName", "DriverNumber", "Time", "Position"])
