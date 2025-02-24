# WasteWise
WasteWise is an image dataset and machine learning model designed to classify waste into three categories: recyclable, non-recyclable, and compostable.

## Definitions of Categories: 

### Recyclable Material:
Definition: Recyclable materials are items that can be collected, processed, and remanufactured into new products. These materials are typically diverted from landfills to be reused.
Examples: Paper, cardboard, glass bottles, aluminum cans, certain plastics (like PETE #1 and HDPE #2).

### Non-Recyclable Material:
Definition: Non-recyclable materials are items that cannot be processed effectively in standard recycling systems, often due to contamination, material composition, or the lack of appropriate recycling infrastructure.
Examples: Styrofoam, plastic bags (unless collected separately), greasy pizza boxes, certain mixed-material packaging, ceramics.

### Compostable Material:
Definition: Compostable materials are organic matter that can break down into nutrient-rich soil through the natural composting process, facilitated by microorganisms, heat, and moisture.
Examples: Fruit and vegetable scraps, coffee grounds, eggshells, yard waste, compostable paper products, certified compostable packaging.


### This dataset was compiled from images retreived from:

**Google API**

**Trashnet dataset (https://github.com/garythung/trashnet?tab=readme-ov-file )**

**Drinking waste classification**

- 4832 images
- sorted by dir
- 4 classes of drinking waste: Aluminium Cans, Glass bottles, PET (plastic) bottles and HDPE (plastic) Milk bottles.
- All images in this dataset are recyclable 

**Taco Dataset**

- 1530 images 
-  It contains images of litter taken under diverse environments: woods, roads and beaches.
- raw images and annotation json
- need to get the categories of trash correct to sort

### Standardization and Augmentation: 

**Manipulations: 14 different manipulations**
-  gray scale
- rot 90, 180, 270
- horizontal flip, vert flip
- noise
- blur
- brighten
- darken
- invert colors
- posterize
- solarize
- equalize 

**Standardize**

- images -> 224x224 numpy arrays or torch tensors

### Images are loaded into Hugging Face