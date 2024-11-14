# Read the data from the file
def read_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.readlines()

# Process the data to separate apartments and addresses
def process_data(lines):
    apartments = []
    addresses = []
    current_page_apartments = []
    current_page_addresses = []

    for line in lines:
        line = line.strip()
        if line.startswith('--- Page'):
            # Store the current page data before moving to the next page
            if current_page_apartments:
                apartments.append(current_page_apartments)
                addresses.append(current_page_addresses)
                current_page_apartments = []
                current_page_addresses = []
        elif line.startswith(('Wohnung zum Kauf', 'Wohnung zur Zwangsversteigerung', 'Penthouse zum Kauf', 'Studio zum Kauf', 'Maisonette zum Kauf', 'Terrassenwohnung zum Kauf')):
            current_page_apartments.append(line)
        elif line and not line.startswith(('Wohnung zum Kauf', 'Wohnung zur Zwangsversteigerung', 'Penthouse zum Kauf', 'Studio zum Kauf', 'Maisonette zum Kauf', 'Terrassenwohnung zum Kauf')):
            current_page_addresses.append(line)

    # Append the last page data if exists
    if current_page_apartments:
        apartments.append(current_page_apartments)
        addresses.append(current_page_addresses)

    return apartments, addresses

# Match addresses with apartments
def match_addresses(apartments, addresses):
    matched_data = []
    for page_apartments, page_addresses in zip(apartments, addresses):
        matched_page = []
        for apartment in page_apartments:
            matched_page.append((apartment, page_addresses.pop(0) if page_addresses else "No address"))
        matched_data.append(matched_page)
    return matched_data

# Output the matched data to a file
def output_matched_data_to_file(matched_data, output_file_path):
    with open(output_file_path, 'w', encoding='utf-8') as file:
        for page_index, page in enumerate(matched_data):
            file.write(f"--- Page {page_index + 1} ---\n")
            for apartment, address in page:
                file.write(f"Apartment: {apartment} | Address: {address}\n")
            file.write("\n")

# Main function to run the script
def main():
    input_file_path = 'data-appartement.txt'  # Update with your actual input file path
    output_file_path = 'data-appartement-matched.txt'  # Output file path
    lines = read_data(input_file_path)
    apartments, addresses = process_data(lines)
    matched_data = match_addresses(apartments, addresses)
    output_matched_data_to_file(matched_data, output_file_path)

if __name__ == "__main__":
    main()
    