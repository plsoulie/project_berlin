import re
import pandas as pd

# Load data from a text file
with open("/Users/willem/Desktop/project_berlin/data_1_scrapping.txt", "r", encoding="utf-8") as file:
    raw_data = file.readlines()

# Initialize lists to store structured data
property_type, location, price, rooms, area, floor, availability, land_area = [], [], [], [], [], [], [], []

# Regular expressions to capture different data points
price_pattern = re.compile(r"(\d+\.?\d*) €")
rooms_pattern = re.compile(r"(\d+,\d*) m²")
floor_pattern = re.compile(r"(\d+)\. Geschoss")
land_area_pattern = re.compile(r"(\d+,\d*) m² Grundstück")
availability_pattern = re.compile(r"frei ab (\w+)")

# Parse each entry in the raw data
for entry in raw_data:
    # Split by ' - ' to separate main fields
    parts = entry.strip().split(" - ")
    
    # Assign property type and location directly
    property_type.append(parts[0])
    location.append(parts[1])
    
    # Extract price
    price_match = price_pattern.search(entry)
    price.append(float(price_match.group(1).replace(".", "")) if price_match else None)
    
    # Extract number of rooms
    rooms_match = re.search(r"(\d+(\.\d+)? Zimmer)", entry)
    rooms.append(rooms_match.group(1).split()[0] if rooms_match else None)
    
    # Extract area in square meters
    area_match = rooms_pattern.search(entry)
    area.append(float(area_match.group(1).replace(",", ".")) if area_match else None)
    
    # Extract floor
    floor_match = floor_pattern.search(entry)
    floor.append(floor_match.group(1) if floor_match else None)
    
    # Extract land area (if applicable)
    land_area_match = land_area_pattern.search(entry)
    land_area.append(float(land_area_match.group(1).replace(",", ".")) if land_area_match else None)
    
    # Extract availability
    availability_match = availability_pattern.search(entry)
    availability.append(availability_match.group(1) if availability_match else "Unknown")

# Create DataFrame
data = {
    "Property Type": property_type,
    "Location": location,
    "Price (€)": price,
    "Rooms": rooms,
    "Area (m²)": area,
    "Floor": floor,
    "Availability": availability,
    "Land Area (m²)": land_area
}

df = pd.DataFrame(data)

# Save the DataFrame to an Excel file
output_file = "/Users/willem/Desktop/project_berlin/cleaned_real_estate_data.xlsx"
df.to_excel(output_file, index=False)

print(f"\nData has been saved to {output_file}")
