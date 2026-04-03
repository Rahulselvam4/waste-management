# utils/disposal_info.py

DISPOSAL_DATA = {
    "cardboard": {
        "emoji": "📦", "color": "#92400e", "bg": "#fef3c7", "accent": "#f59e0b",
        "bin": "Blue Recycling Bin", "action": "Recycle",
        "tip": "Flatten boxes and keep them dry.",
        "steps": [
            "Remove all tape, staples and plastic wrapping",
            "Flatten the box completely to save bin space",
            "Keep dry — wet cardboard cannot be recycled",
            "Place in blue recycling bin or recycling centre",
            "Greasy pizza boxes go in general waste",
        ],
        "fact": "Recycling one tonne of cardboard saves 17 trees and 7,000 gallons of water.",
        "recyclable": True, "co2": "1.1 kg CO₂ saved/kg", "decompose": "2 months", "rate": "89%",
    },
    "glass": {
        "emoji": "🫙", "color": "#1e3a5f", "bg": "#dbeafe", "accent": "#3b82f6",
        "bin": "Green/Brown Bottle Bank", "action": "Recycle",
        "tip": "Rinse and sort by colour.",
        "steps": [
            "Rinse bottles and jars to remove food residue",
            "Remove metal lids — recycle them separately",
            "Do NOT include window glass, mirrors or Pyrex",
            "Separate by colour at bottle banks if required",
            "Broken glass: wrap in newspaper, put in general waste",
        ],
        "fact": "Glass can be recycled indefinitely without any loss in quality or purity.",
        "recyclable": True, "co2": "0.3 kg CO₂ saved/kg", "decompose": "1 million years", "rate": "67%",
    },
    "metal": {
        "emoji": "🥫", "color": "#1f2937", "bg": "#f3f4f6", "accent": "#6b7280",
        "bin": "Blue Recycling Bin", "action": "Recycle",
        "tip": "Rinse cans and crush to save space.",
        "steps": [
            "Rinse food and drink cans thoroughly",
            "Crush cans to save space in the bin",
            "Aluminium foil: scrunch into a ball first",
            "Aerosol cans: ensure completely empty",
            "Remove paper labels where possible",
        ],
        "fact": "Recycling aluminium uses only 5% of the energy needed to produce new aluminium.",
        "recyclable": True, "co2": "9.1 kg CO₂ saved/kg", "decompose": "200–500 years", "rate": "70%",
    },
    "paper": {
        "emoji": "📄", "color": "#14532d", "bg": "#dcfce7", "accent": "#22c55e",
        "bin": "Blue Recycling Bin", "action": "Recycle",
        "tip": "Keep dry — wet paper cannot be recycled.",
        "steps": [
            "Keep paper clean and dry before recycling",
            "OK: newspapers, magazines, office paper, envelopes",
            "NOT OK: tissue, paper towels, waxed paper",
            "Shredded paper: place in a sealed paper bag",
            "Remove plastic windows from envelopes first",
        ],
        "fact": "Each tonne of recycled paper saves 24 trees and 7,000 litres of water.",
        "recyclable": True, "co2": "0.9 kg CO₂ saved/kg", "decompose": "2–6 weeks", "rate": "66%",
    },
    "plastic": {
        "emoji": "🧴", "color": "#7c2d12", "bg": "#fff7ed", "accent": "#f97316",
        "bin": "Check Symbol — #1 & #2 Recyclable", "action": "Check Symbol",
        "tip": "Look for the triangle number on the bottom.",
        "steps": [
            "Look for triangle recycling symbol on the bottom",
            "#1 (PET) and #2 (HDPE) are widely accepted",
            "#3–#7: check your local council guidelines",
            "Rinse containers to remove food residue",
            "Plastic bags: return to supermarket drop-off points",
        ],
        "fact": "Only 9% of all plastic ever produced has been recycled globally.",
        "recyclable": True, "co2": "1.5 kg CO₂ saved/kg", "decompose": "450+ years", "rate": "9%",
    },
    "trash": {
        "emoji": "🗑️", "color": "#374151", "bg": "#f9fafb", "accent": "#9ca3af",
        "bin": "Black General Waste Bin", "action": "General Waste",
        "tip": "Reduce future waste by choosing less packaging.",
        "steps": [
            "Place in black general waste bin",
            "Check if any parts can be separated and recycled",
            "Hazardous items: take to special collection points",
            "E-waste: take to dedicated e-waste collection",
            "Consider composting organic components",
        ],
        "fact": "The average person generates over 4 lbs of trash daily. Small changes add up.",
        "recyclable": False, "co2": "Reduce consumption", "decompose": "Varies", "rate": "0%",
    },
}

GLOBAL_STATS = [
    {"label": "Waste generated annually",  "value": "2.01B tonnes",   "icon": "🌍"},
    {"label": "Global recycling rate",      "value": "Only 19%",       "icon": "♻️"},
    {"label": "Plastic enters oceans/year", "value": "8M tonnes",      "icon": "🌊"},
    {"label": "Landfill methane share",     "value": "11% of global",  "icon": "☁️"},
]
