# rebuild
with open("diagram 2.py", "w") as f:
    f.write("""# %%
# Annotated audio sample showing different types of annotation
import librosa
import librosa.display
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Set global font settings to match sankey
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 16,
    'text.color': 'black'
})

lyrics_1 = "Is death real?"
lyrics_2 = "mmm, you're gonna want me back in your arms"

audio_path = "J Dilla - Stop - Donuts Full Album [2026-03-29 173144].wav"
y, sr = librosa.load(audio_path, duration=12)

print(f"Audio loaded with shape {y.shape} and sample rate {sr}")

# Horizontal layout - wide aspect ratio
fig, ax = plt.subplots(figsize=(24, 14))
librosa.display.waveshow(y, sr=sr, ax=ax, alpha=0.5, color='#21386a')

# Set figure background colors to white
fig.patch.set_facecolor('white')
ax.set_facecolor('white')

# Adjust limits for side-by-side layout
ax.set_ylim(bottom=-12.0, top=6.0)
ax.set_xlim(left=-2.0, right=28.0)
ax.axis('off')

# --- MAIN TITLE ---
fig.suptitle("Relational Context in Musical Meaning\\nJ Dilla \u2013 'Stop!' (2006)", fontsize=26, fontweight='bold', color="#21386a", va='top', y=0.98)

# --- 1) CONTENT CLASS FRAME (LEFT) ---
content_box = patches.Rectangle((-1.5, -7.0), 14.5, 12.0, linewidth=2, edgecolor='#21386a', facecolor='#f9f9f9', alpha=0.5, linestyle='solid')
ax.add_patch(content_box)
ax.text(-2.0, -1.0, "Class: Content (Audio & Lyrics)", fontsize=22, fontweight='bold', va='center', ha='center', color='#21386a', rotation=90)

# Calculate musical features
tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
t_val = float(tempo[0] if isinstance(tempo, np.ndarray) else tempo)
chroma = librosa.feature.chroma_stft(y=y, sr=sr)
estimated_key = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'][np.argmax(np.sum(chroma, axis=1))]

# Sankey colors matching diagram 1
sankey_colors = {
    'genre': '#EF553B',     # Orange/Red
    'timbre': '#00CC96',    # Green
    'structure': '#AB63FA'  # Purple
}

ax.text(2.0, 4.0, "Genre:\\nInstrumental Hip-Hop", fontsize=12, fontweight='bold', color='white', ha='center', va='center',
        bbox=dict(boxstyle="round,pad=0.7", fc=sankey_colors['genre'], ec="none", alpha=0.9))

import timbral_models
h_val = timbral_models.timbral_hardness(audio_path)
v_val = timbral_models.timbral_warmth(audio_path)
b_val = timbral_models.timbral_brightness(audio_path)

timbral_str = f"Timbral Descriptors:\\nWarm: {v_val:.0f} | Hard: {h_val:.0f} | Bright: {b_val:.0f}"
ax.text(6.0, 4.0, timbral_str, fontsize=12, fontweight='bold', color='white', ha='center', va='center',
        bbox=dict(boxstyle="round,pad=0.7", fc=sankey_colors['timbre'], ec="none", alpha=0.9))

musical_info = f"Musical Structure:\\n{t_val:.0f} BPM, {estimated_key} Minor"
ax.text(10.5, 4.0, musical_info, fontsize=12, fontweight='bold', color='white', ha='center', va='center',
        bbox=dict(boxstyle="round,pad=0.7", fc=sankey_colors['structure'], ec="none", alpha=0.9))

ai_caption = "AI Caption (WavCaps):\\n'A repeating soul vocal sample over\\na dusty, lo-fi boom-bap drum beat.'"
ax.text(6.0, -5.0, ai_caption, fontsize=16, color='#2ca02c', ha='center', va='center', style='italic', 
        bbox=dict(boxstyle="round,pad=0.5", fc="#e6ffe6", ec="#2ca02c", lw=1.5, alpha=0.9))

# Annotate lyrics with different colors underneath the waveform
ax.text(1.5, -2.0, lyrics_1, fontsize=18, color='#d62728', ha='center', va='center', fontweight='bold', bbox=dict(facecolor='white', edgecolor='none', alpha=0.8, pad=0))
ax.text(8.5, -3.0, lyrics_2, fontsize=18, color='#1f77b4', ha='center', va='center', fontweight='bold', bbox=dict(facecolor='white', edgecolor='none', alpha=0.8, pad=0))
ax.text(10.5, -2.0, lyrics_1, fontsize=18, color='#d62728', ha='center', va='center', fontweight='bold', bbox=dict(facecolor='white', edgecolor='none', alpha=0.8, pad=0))

# Lyric highlight spans across waveform
y_range = 18.0
base_pct = 12.0 / y_range 
span_min = base_pct - (1.0/y_range)
span_max = base_pct + (1.0/y_range)

ax.axvspan(0.0, 3.0, ymin=span_min, ymax=span_max, color='#d62728', alpha=0.15)
ax.axvspan(6.0, 11.0, ymin=span_min, ymax=span_max, color='#1f77b4', alpha=0.15)
ax.axvspan(9.0, 12.0, ymin=span_min, ymax=span_max, color='#d62728', alpha=0.15)

# --- 2) CONTEXT CLASS FRAME (RIGHT) ---
context_box = patches.Rectangle((13.5, -7.0), 14.0, 12.0, linewidth=2, edgecolor='#21386a', facecolor='#f0f0f0', alpha=0.5, linestyle='solid')
ax.add_patch(context_box)
ax.text(13.0, -1.0, "Class: Context\\n(Relational Metadata)", fontsize=22, fontweight='bold', va='center', ha='center', color='#21386a', rotation=90)

# Context Node 2 (Jadakiss) - mapping to lyrics 1 at x=1.5
ax.annotate("Sample Origin:\\nJadakiss feat. Anthony Hamilton 'Why?'\\nOriginal Lyric: 'it\\s that real...'", 
            xy=(3.0, -1.0), xycoords='data',
            xytext=(18.0, 1.5), textcoords='data',
            arrowprops=dict(arrowstyle="->,head_width=1.0,head_length=1.2", connectionstyle="arc3,rad=0.1", color='#d62728', lw=2.5),
            bbox=dict(boxstyle="round,pad=0.8", fc="#ffe6e6", ec="#d62728", lw=2),
            ha='center', va='center', fontsize=16, fontweight='semibold')

# Context Node 1 (Dionne Warwick) - mapping to lyrics 2 at x=8.5
ax.annotate("Sample Origin:\\nDionne Warwick\\n'You\\re Gonna Need Me'", 
            xy=(11.0, -3.0), xycoords='data',
            xytext=(18.0, -3.5), textcoords='data',
            arrowprops=dict(arrowstyle="->,head_width=1.0,head_length=1.2", connectionstyle="arc3,rad=-0.1", color='#1f77b4', lw=2.5),
            bbox=dict(boxstyle="round,pad=0.8", fc="#e6f2ff", ec="#1f77b4", lw=2),
            ha='center', va='center', fontsize=16, fontweight='semibold')

# --- 3) PRODUCTION CONTEXT (NESTED WITHIN CONTEXT) ---
production_ctx_box = patches.Rectangle((22.0, -5.5), 5.0, 9.0, linewidth=2.0, edgecolor='#9467bd', facecolor='#e8e8e8', alpha=0.6, linestyle='--')
ax.add_patch(production_ctx_box)
ax.text(24.5, 2.0, "Subclass:\\nProduction\\nContext", fontsize=16, fontweight='bold', va='center', ha='center', color='#9467bd')

# Meta Context Node
meta_text = "Produced By:\\nJ Dilla\\nAlbum: Donuts (2006)\\n\\nProduced during\\nhis final days battling\\nlupus and TTP"
ax.text(24.5, -1.5, meta_text, fontsize=14, fontweight='semibold', color='#9467bd', ha='center', va='center',
        bbox=dict(boxstyle="round,pad=0.8", fc="#f2e6ff", ec="#9467bd", lw=2))

# --- 4) INTERPRETATION LAYER (BOTTOM ROW) ---
interpretation_box = patches.Rectangle((-1.5, -11.0), 29.0, 3.5, linewidth=2, edgecolor='#21386a', facecolor='#f5f5dc', alpha=0.5, linestyle='solid')
ax.add_patch(interpretation_box)
ax.text(-2.0, -9.25, "Interpretation\\nSynthesis", fontsize=22, fontweight='bold', va='center', ha='center', color='#21386a', rotation=90)

interpretation_text = "Meaning: In his final days, Dilla used aural illusion to transform an ad-lib into a parting reflection on his own mortality."
ax.text(13.0, -9.25, interpretation_text, fontsize=20, color='black', ha='center', va='center', style='italic',
        bbox=dict(boxstyle="round,pad=1.0", fc="#ffffff", ec="#21386a", lw=2, alpha=0.9))

plt.subplots_adjust(top=0.92, bottom=0.05, left=0.08, right=0.98)
# plt.show()
""")
