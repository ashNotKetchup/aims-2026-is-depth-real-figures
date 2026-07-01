import timbral_models
audio_path = "J Dilla - Stop - Donuts Full Album [2026-03-29 173144].wav"
hardness = timbral_models.timbral_hardness(audio_path)
warmth = timbral_models.timbral_warmth(audio_path)
brightness = timbral_models.timbral_brightness(audio_path)
print(f"Hardness: {hardness}")
print(f"Warmth: {warmth}")
print(f"Brightness: {brightness}")
