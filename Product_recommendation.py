import re

# Function to generate product variants dynamically with extended suffixes and product links
def generate_product_variants(product_name):
    # Common suffixes for various product types
    suffixes = {
        "phone": [
            "", "Pro", "Pro Max", "Max", "Mini", "Plus", "SE", "Ultra", "5G", "Lite", "X", "XL", "Neo", "S", "A", "Z", "Classic"
        ],
        "tv": [
            "Smart", "LED", "OLED", "QLED", "UHD", "4K", "8K", "HDR", "Slim", "Ultra HD", "Full HD", "Curved", "Flat"
        ],
        "laptop": [
            "Air", "Pro", "Ultra", "Plus", "Max", "2-in-1", "Gaming", "Touch", "Convertible", "Business", "Edition", "Workstation"
        ],
        "accessory": [
            "Wireless", "Bluetooth", "Wired", "Charging", "Case", "Sleeve", "Stand", "Keyboard", "Mouse", "Cable", "Docking"
        ],
        "general": [
            "Basic", "Standard", "Advanced", "Deluxe", "Elite", "Edition", "Special", "Premium", "Limited", "New", "2023", "2024"
        ]
    }
    
    # Define a set of keywords to categorize the product name into types
    categories = {
        "phone": ["phone", "smartphone", "iphone", "android", "cellphone"],
        "tv": ["tv", "television", "led", "oled", "qled"],
        "laptop": ["laptop", "notebook", "macbook", "chromebook", "ultrabook"],
        "accessory": ["case", "charger", "headphone", "earphone", "keyboard", "mouse", "cable", "speaker"]
    }

    # Determine the category of the product based on the keywords in the product name
    product_type = "general"  # Default category
    for category, keywords in categories.items():
        if any(keyword.lower() in product_name.lower() for keyword in keywords):
            product_type = category
            break
    
    # Fetch appropriate suffixes for the determined product category
    selected_suffixes = suffixes.get(product_type, suffixes["general"])

    # Generate product variants by appending each suffix to the base product name
    variants = [f"{product_name} {suffix}".strip() for suffix in selected_suffixes]
    
    # Remove duplicate variants (if any) using set
    unique_variants = list(set(variants))

    # Generate hypothetical product URLs based on the product name (example)
    product_links = [f"https://www.amazon.com/s?k={product_name.replace(' ', '+')}{suffix.replace(' ', '+')}" for suffix in selected_suffixes]
    
    return unique_variants, product_links

# Main function to process input and generate variants with links
'''
def main(product_name):
    try:
        # Generate variants and links for the given product
        variants, links = generate_product_variants(product_name)

        # Output the variants and their respective links
        for variant, link in zip(variants, links):
            print(f"Variant: {variant}\nLink: {link}\n")

    except Exception as e:
        print(f"Error: {e}")

# Run the script with an example product name
if __name__ == "__main__":
    product_name = input("enter product")  # Example input
    main(product_name)
    '''
