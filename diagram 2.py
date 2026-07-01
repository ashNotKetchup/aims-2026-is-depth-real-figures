# %%
# Annotated audio sample showing different types of annotation
import librosa
import librosa.display
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import warnings
import timbral_models
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

# Blocky layout
fig, ax = plt.subplots(figsize=(24, 16))
librosa.display.waveshow(y, sr=sr, ax=ax, alpha=0.5, color='#21386a')

# Set figure background colors to white
fig.patch.set_facecolor('white')
ax.set_facecolor('white')

# Adjust limits for the new blocky layout (extended x-axis for right-side boxes)
ax.set_ylim(bottom=-18.0, top=6.0)
ax.set_xlim(left=-2.0, right=28.0)
ax.axis('off')

# --- MAIN TITLE ---
fig.suptitle("Relational Context in Musical Meaning\nJ Dilla – 'Stop!' (2006)", fontsize=26, fontweight='bold', color="#21386a", va='top', y=0.98)

# --- 1) CONTENT CLASS FRAME (TOP, EXPANDED TO RIGHT) ---
content_box = patches.Rectangle((-1.5, -7.0), 29.0, 12.5, linewidth=2, edgecolor='#21386a', facecolor='#f9f9f9', alpha=0.5, linestyle='solid')
ax.add_patch(content_box)
ax.text(-2.0, -1.0, "Class: Content\n(Audio & Lyrics)", fontsize=22, fontweight='bold', va='center', ha='center', color='#21386a', rotation=90)

# Calculate musical features
tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
t_val = float(tempo[0] if isinstance(tempo, np.ndarray) else tempo)
chroma = librosa.feature.chroma_stft(y=y, sr=sr)
estimated_key = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'][np.argmax(np.sum(chroma, axis=1))]

# Sankey colors matching diagram 1
sankey_colors = {
    'genre': '#EF553B',     
    'timbre': '#00CC96',    
    'musical structure': '#AB63FA',  
    'text': '#1f77b4'
}

# Stacked annotation block on the right (like a highlighted table)
text_x = 13.0
box_style = dict(boxstyle="square,pad=0.3", ec="white", lw=1, alpha=0.9)
row_font_size = 11

# Row 1: Genre
ax.text(text_x, 4.0, "Genre: Instrumental Hip-Hop", fontsize=row_font_size, fontweight='bold', color='white', ha='left', va='center',
        bbox=dict(**box_style, fc=sankey_colors['genre']))

# Row 2: Timbre
h_val = timbral_models.timbral_hardness(audio_path)
v_val = timbral_models.timbral_warmth(audio_path)
b_val = timbral_models.timbral_brightness(audio_path)

timbral_str = f"Timbral Descriptors: Warm {v_val:.0f} | Hard {h_val:.0f} | Bright {b_val:.0f}"
ax.text(text_x, 3.7, timbral_str, fontsize=row_font_size, fontweight='bold', color='white', ha='left', va='center',
        bbox=dict(**box_style, fc=sankey_colors['timbre']))

# Row 3: Musical Structure
musical_info = f"Musical Structure: {t_val:.0f} BPM, {estimated_key} Minor"
ax.text(text_x, 2.4, musical_info, fontsize=row_font_size, fontweight='bold', color='white', ha='left', va='center',
        bbox=dict(**box_style, fc=sankey_colors['musical structure']))

# Row 4: Textual Description / AI Caption
ai_caption = "AI Caption: A repeating soul vocal sample\nover a dusty, lo-fi boom-bap drum beat."
ax.text(text_x, -0.2, ai_caption, fontsize=row_font_size, fontweight='bold', color='white', ha='left', va='center', 
        bbox=dict(**box_style, fc=sankey_colors['text']))

# Annotate lyrics with different colors underneath the waveform (kept under the waveform on the left)
ax.text(1.5, -2.0, lyrics_1, fontsize=18, color='#d62728', ha='center', va='center', fontweight='bold', bbox=dict(facecolor='white', edgecolor='none', alpha=0.8, pad=0))
ax.text(8.5, -3.0, lyrics_2, fontsize=18, color='#1f77b4', ha='center', va='center', fontweight='bold', bbox=dict(facecolor='white', edgecolor='none', alpha=0.8, pad=0))
ax.text(10.5, -2.0, lyrics_1, fontsize=18, color='#d62728', ha='center', va='center', fontweight='bold', bbox=dict(facecolor='white', edgecolor='none', alpha=0.8, pad=0))

# Lyric highlight spans across waveform
# The y-axis ranges from -18.0 to 6.0 (total range 24.0)
y_range_total = 24.0
span_min = (-1.2 - (-18.0)) / y_range_total
span_max = (1.2 - (-18.0)) / y_range_total

ax.axvspan(0.0, 3.0, ymin=span_min, ymax=span_max, color='#d62728', alpha=0.15)
ax.axvspan(6.0, 11.0, ymin=span_min, ymax=span_max, color='#1f77b4', alpha=0.15)
ax.axvspan(9.0, 12.0, ymin=span_min, ymax=span_max, color='#d62728', alpha=0.15)

# --- 2) CONTEXT CLASS FRAME (BOTTOM) ---
context_box = patches.Rectangle((-1.5, -17.5), 29.0, 9.5, linewidth=2, edgecolor='#21386a', facecolor='#f0f0f0', alpha=0.5, linestyle='solid')
ax.add_patch(context_box)
ax.text(-2.0, -12.75, "Class: Context\n(Relational Metadata)", fontsize=22, fontweight='bold', va='center', ha='center', color='#21386a', rotation=90)

# Context Node 2 (Jadakiss) - mapping to lyrics 1 at x=1.5
ax.annotate("Sample Origin:\nJadakiss feat. Anthony Hamilton 'Why?'\nOriginal Lyric: 'it's that real...'", 
            xy=(2.0, -2.5), xycoords='data',
            xytext=(4.0, -12.5), textcoords='data',
            arrowprops=dict(arrowstyle="->,head_width=1.0,head_length=1.2", connectionstyle="arc3,rad=0.1", color='#d62728', lw=2.5),
            bbox=dict(boxstyle="round,pad=0.8", fc="#ffe6e6", ec="#d62728", lw=2),
            ha='center', va='center', fontsize=14, fontweight='semibold')

# Context Node 1 (Dionne Warwick) - mapping to lyrics 2 at x=8.5
ax.annotate("Sample Origin:\nDionne Warwick\n'You're Gonna Need Me'", 
            xy=(8.5, -3.5), xycoords='data',
            xytext=(12.0, -12.5), textcoords='data',
            arrowprops=dict(arrowstyle="->,head_width=1.0,head_length=1.2", connectionstyle="arc3,rad=-0.1", color='#1f77b4', lw=2.5),
            bbox=dict(boxstyle="round,pad=0.8", fc="#e6f2ff", ec="#1f77b4", lw=2),
            ha='center', va='center', fontsize=14, fontweight='semibold')

# --- 3) PRODUCTION CONTEXT (NESTED WITHIN CONTEXT) ---
production_ctx_box = patches.Rectangle((18.0, -16.5), 9.0, 7.5, linewidth=2.0, edgecolor='#9467bd', facecolor='#e8e8e8', alpha=0.6, linestyle='--')
ax.add_patch(production_ctx_box)
ax.text(19.0, -12.75, "Subclass:\nProduction\nContext", fontsize=14, fontweight='bold', va='center', ha='center', color='#9467bd', rotation=90)

# Meta Context Node
meta_text = "Produced By:\nJ Dilla\nAlbum: Donuts (2006)\n\nProduced during his\nfinal days battling\nlupus and TTP"
ax.text(23.5, -12.75, meta_text, fontsize=14, fontweight='semibold', color='#9467bd', ha='center', va='center',
        bbox=dict(boxstyle="round,pad=0.8", fc="#f2e6ff", ec="#9467bd", lw=2))

plt.subplots_adjust(top=0.92, bottom=0.05, left=0.10, right=0.98)
plt.show()
