"""Short, farmer-facing guidance per PlantVillage class (general agronomy
practice – always confirm with a local Krishi Vigyan Kendra / extension officer)."""

ADVICE = {
    "scab": "Remove fallen leaves, prune for airflow; protective fungicide (captan / mancozeb) from bud break.",
    "Black_rot": "Remove mummified fruit and cankers; sanitation pruning; fungicide during wet weather.",
    "Cedar_apple_rust": "Remove nearby juniper galls if possible; myclobutanil-type fungicide at pink bud stage.",
    "Powdery_mildew": "Improve airflow, avoid excess nitrogen; sulphur or potassium-bicarbonate sprays.",
    "Cercospora_leaf_spot Gray_leaf_spot": "Rotate crops, bury residue; resistant hybrids; strobilurin fungicide if severe.",
    "Common_rust_": "Plant resistant hybrids; fungicide only if infection appears before tasseling.",
    "Northern_Leaf_Blight": "Resistant hybrids, crop rotation, residue management; fungicide at early lesions.",
    "Esca_(Black_Measles)": "Prune out infected wood in dry weather, protect pruning wounds; no curative spray.",
    "Leaf_blight_(Isariopsis_Leaf_Spot)": "Remove infected leaves; copper or mancozeb sprays after rain.",
    "Haunglongbing_(Citrus_greening)": "No cure – remove infected trees, control the psyllid vector, use certified saplings.",
    "Bacterial_spot": "Use disease-free seed, avoid overhead irrigation; copper sprays reduce spread.",
    "Early_blight": "Remove lower infected leaves, mulch, rotate 2-3 years; chlorothalonil / mancozeb.",
    "Late_blight": "URGENT – spreads in days. Destroy infected plants; metalaxyl/mancozeb or cymoxanil sprays.",
    "Leaf_scorch": "Renovate beds after harvest, remove old leaves; avoid overhead watering.",
    "Leaf_Mold": "Lower greenhouse humidity, increase ventilation; remove infected leaves.",
    "Septoria_leaf_spot": "Remove infected lower leaves, mulch, avoid wetting foliage; chlorothalonil.",
    "Spider_mites Two-spotted_spider_mite": "Spray water on leaf undersides, neem oil or miticide; avoid broad-spectrum insecticides.",
    "Target_Spot": "Improve airflow, remove debris; protective fungicide in humid weather.",
    "Tomato_Yellow_Leaf_Curl_Virus": "No cure – remove infected plants, control whiteflies (yellow sticky traps, nets).",
    "Tomato_mosaic_virus": "No cure – remove plants, disinfect tools and hands; don't use tobacco near plants.",
    "healthy": "No disease detected. Keep monitoring weekly, especially after rain.",
}


def advice_for(cls: str) -> str:
    disease = cls.split("___", 1)[-1]
    if "healthy" in disease:
        return ADVICE["healthy"]
    for key, txt in ADVICE.items():
        if key.lower() in disease.lower() or disease.lower() in key.lower():
            return txt
    return "Consult a local agricultural extension officer for treatment."


def pretty(cls: str) -> tuple[str, str]:
    plant, disease = cls.split("___", 1)
    return plant.replace("_", " ").replace(",", ""), disease.replace("_", " ").strip()
