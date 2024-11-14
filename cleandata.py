import os  # Import the os module
import re
import pandas as pd

# Load data from a text file
with open('data-appartement-matched.txt', "r", encoding="utf-8") as file:
    raw_data = file.readlines()

# Initialize lists to store structured data
property_type, location, price, rooms, area, floor, availability, land_area = [], [], [], [], [], [], [], []
neighborhood, zip_code = [], []  # New lists for neighborhood and zip code

# Regular expressions to capture different data points
price_pattern = re.compile(r"(\d{1,3}(?:\.\d{3})*) €")  # Capture prices with thousands separator
rooms_pattern = re.compile(r"(\d+) Zimmer")  # Capture number of rooms
area_pattern = re.compile(r"(\d+,\d*) m²")  # Capture area in m²
floor_pattern = re.compile(r"(\d+)\. Geschoss")  # Capture floor information
availability_pattern = re.compile(r"frei ab (\w+)")  # Capture availability
address_pattern = re.compile(r"(.+), Berlin \((\d{5})\)")  # Pattern to capture neighborhood and zip code

# Parse each entry in the raw data
for entry in raw_data:
    # Split by ' - ' to separate main fields
    parts = entry.strip().split(" | ")
    
    # Extract property type and location
    if len(parts) > 0:
        property_info = parts[0].split(": ")
        if len(property_info) > 1:
            property_type.append(property_info[1])  # Extract property type
            location.append(parts[1])  # Berlin is the location
        else:
            property_type.append("Unknown")
            location.append("Unknown")
    
    # Extract price
    price_match = price_pattern.search(entry)
    if price_match:
        price.append(price_match.group(0))  # Append the full matched price string
    else:
        price.append("Preis auf Anfrage")  # Handle case for "Preis auf Anfrage"
    
    # Extract rooms
    rooms_match = rooms_pattern.search(entry)
    rooms.append(rooms_match.group(1) if rooms_match else "Unknown")
    
    # Extract area
    area_match = area_pattern.search(entry)
    area.append(area_match.group(1) if area_match else "Unknown")
    
    # Extract floor
    floor_match = floor_pattern.search(entry)
    floor.append(floor_match.group(1) if floor_match else "N/A")  # Use N/A if not specified
    
    # Extract land area (if applicable)
    land_area_match = re.search(r"(\d+,\d*) m² Grundstück", entry)
    land_area.append(land_area_match.group(1) if land_area_match else "Unknown")
    
    # Extract availability
    availability_match = availability_pattern.search(entry)
    availability.append(availability_match.group(1) if availability_match else "Unknown")
    
    # Extract neighborhood and zip code
    address_match = address_pattern.search(entry)
    if address_match:
        neighborhood.append(address_match.group(1).strip())  # Neighborhood
        zip_code.append(address_match.group(2).strip())  # Zip code
    else:
        neighborhood.append("Unknown")
        zip_code.append("Unknown")

# Create DataFrame
data = {
    "Property Type": property_type,
    "Location": location,
    "Price (€)": price,
    "Rooms": rooms,
    "Area (m²)": area,
    "Floor": floor,
    "Availability": availability,
    "Land Area (m²)": land_area,
    "Neighborhood": neighborhood,  # New column for neighborhood
    "Zip Code": zip_code  # New column for zip code
}

df = pd.DataFrame(data)

# Define output file paths
output_file_xlsx = 'cleaned_real_estate_data.xlsx'
output_file_csv = 'cleaned_real_estate_data.csv'

# Remove existing CSV file if it exists
if os.path.exists(output_file_csv):
    os.remove(output_file_csv)

# Save the DataFrame to an Excel file
df.to_excel(output_file_xlsx, index=False)

# Save the DataFrame to a CSV file
df.to_csv(output_file_csv, index=False)

print(f"\nData has been saved to {output_file_xlsx} and {output_file_csv}")
