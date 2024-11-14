import os
import re
import pandas as pd

# Load data from a text file
with open('data-appartement-matched.txt', "r", encoding="utf-8") as file:
    raw_data = file.readlines()

# Initialize lists to store structured data
property_type, location, price, surface, rooms, floor, zip_code, neighborhood, address = [], [], [], [], [], [], [], [], []

# Property type mapping
property_type_mapping = {
    "Wohnung zum Kauf": "Apartment",
    "Maisonette zum Kauf": "House",
    "Penthouse zum Kauf": "Penthouse",
    "Terrassenwohnung zum Kauf": "Apartment",
    "Studio zum Kauf": "Studio"
}

# Regular expressions to capture different data points
price_pattern = re.compile(r"(\d{1,3}(?:\.\d{3})*|\d+)(?:\s*€)")  # Capture prices with or without thousands separator
rooms_pattern = re.compile(r"(\d+)(?: Zimmer|, \d+ Zimmer)")  # Capture number of rooms
surface_pattern = re.compile(r"(\d+(?:,\d+)?)\s*m²")  # Capture area in m² (integers and decimals)
floor_pattern = re.compile(r"(\d+)\. Geschoss|EG")  # Capture floor information
address_pattern = re.compile(r"Address: (.+) \((\d{5})\)")  # Pattern to capture address and zip code

# Parse each entry in the raw data
for entry in raw_data:
    # Skip lines that do not contain apartment data
    if "Apartment:" not in entry:
        continue

    # Extract property type and location
    property_info = entry.split(" - ")
    if len(property_info) > 0:
        original_property_type = property_info[0].split(": ")[1].strip()  # Extract property type
        # Replace property type using the mapping
        property_type.append(property_type_mapping.get(original_property_type, original_property_type))  # Use mapped value or original
        location.append(property_info[1])  # Berlin is the location

    # Extract price
    price_match = price_pattern.search(entry)
    if price_match:
        price.append(price_match.group(0).strip())  # Append the full matched price string
    else:
        price.append("Preis auf Anfrage")  # Handle case for "Preis auf Anfrage"

    # Extract rooms
    rooms_match = rooms_pattern.search(entry)
    rooms.append(rooms_match.group(1) if rooms_match else "Unknown")

    # Extract surface area
    surface_match = surface_pattern.search(entry)
    surface.append(surface_match.group(1) if surface_match else "Unknown")  # Capture the area value

    # Extract floor
    floor_match = floor_pattern.search(entry)
    floor.append(floor_match.group(1) if floor_match else "N/A")  # Use N/A if not specified

    # Extract address and zip code
    address_match = address_pattern.search(entry)
    if address_match:
        address.append(address_match.group(1).strip())  # Address
        zip_code.append(address_match.group(2).strip())  # Zip code
        
        # Extract neighborhood from address
        address_parts = address_match.group(1).split(",")  # Split address by comma
        if len(address_parts) == 3:
            neighborhood_name = address_parts[1].strip()  # Get the second element
        elif len(address_parts) == 2:
            neighborhood_name = address_parts[0].strip()  # Get the first element if only two parts
        else:
            neighborhood_name = "Unknown"  # Default value if address format is unexpected
        neighborhood.append(neighborhood_name)
    else:
        address.append("Unknown")
        zip_code.append("Unknown")
        neighborhood.append("Unknown")  # Default value if address is unknown

# Create DataFrame with Address at the end
data = {
    "Property Type": property_type,
    "Location": location,
    "Price (€)": price,
    "Surface (m²)": surface,
    "Rooms": rooms,
    "Floor": floor,
    "Zip Code": zip_code,
    "Neighborhood": neighborhood,  # Add the new Neighborhood column
    "Address": address  # Move Address to the end
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