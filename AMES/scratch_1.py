from mendeleev import element

# List of elements to check
elements = ["N", "C", "H", "O", "S", "Cl", "Be", "Br", "Pt", "P",
            "F", "As", "Hg", "Zn", "Si", "V", "I", "B", "Sn", "Ge",
            "Ag", "Sb", "Cu", "Cr", "Pb", "Mo", "Se", "Al", "Cd",
            "Mn", "Fe", "Ga", "Pd", "Na", "Ti", "Bi", "Co", "Ni",
            "Ce", "Ba", "Zr", "Rh"]

# Get all attributes of an element object
sample_element = element("C")
all_attributes = dir(sample_element)

# Filter attributes that are not methods or private variables
valid_attributes = [attr for attr in all_attributes if not attr.startswith("_") and not callable(getattr(sample_element, attr))]

# Check which attributes have no missing values for all elements
valid_keywords = []
for attr in valid_attributes:
    if all(getattr(element(el), attr) is not None for el in elements):
        valid_keywords.append(attr)

print(valid_keywords)