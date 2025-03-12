import numpy as np
import matplotlib
matplotlib.use('Agg')  # שימוש ב-backend ללא GUI
import matplotlib.pyplot as plt
from scipy.io import wavfile
import base64
import io
from flask import Flask, render_template
from sklearn.cluster import KMeans
from scipy.signal import find_peaks
from scipy.signal import stft, istft
import pandas as pd
from scipy.io.wavfile import write

app = Flask(__name__)

# פונקציה לחלוקת האות לחלונות חופפים
def split_signal_into_frames(signal, sample_rate, frame_duration=0.02, overlap=0.25):
    # גודל חלון במונחי דגימות
    frame_size = int(frame_duration * sample_rate)
    # מספר הדגימות להסטה בין חלון לחלון
    step_size = int(frame_size * (1 - overlap))
    # יצירת רשימה של חלונות חופפים
    frames = [
        signal[i:i + frame_size]
        for i in range(0, len(signal) - frame_size + 1, step_size)
    ]
    return frames

# נתיב לדף הראשי
@app.route('/')
def home():
    # קריאת קובץ אודיו
    sample_rate, signal = wavfile.read('C:/Users/yaron/OneDrive/Desktop/Projects/DSP/Python/wave_file (5).wav')
    signal = signal / np.max(np.abs(signal))  # נרמול האות לטווח [-1, 1]
    
    # הצגת פרטי הקובץ
    print(f"Sampling Rate: {sample_rate} Hz")
    print(f"Signal Length: {len(signal)}")

    # יצירת גרף לאות האודיו
    plt.figure(figsize=(10, 4))
    plt.plot(signal)
    plt.title('Audio Signal')
    plt.xlabel('Sample Index')
    plt.ylabel('Amplitude')

    # שמירת הגרף בתמונה Base64
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    graph_url = base64.b64encode(img.getvalue()).decode()

    # חישוב האנרגיה של האות
    energy = np.sum(signal**2) 
    formula = "Energy = Σ(signal^2) "

    return render_template('index.html', graph_url=graph_url, formula=formula, energy=energy, sample_rate=sample_rate, signal1=len(signal))




@app.route('/new_page')
def new_page():
    # קריאת קובץ האודיו
    sample_rate, signal = wavfile.read('C:/Users/yaron/OneDrive/Desktop/Projects/DSP/Python/wave_file (5).wav')
    
    # נרמול האות לטווח [-1, 1]
    signal = signal / np.max(np.abs(signal))
    
    # פרמטרים
    frame_duration = 0.02  # משך כל חלון בשניות
    overlap = 0.25  # חפיפה בין חלונות

    # חישוב גודל חלון והמרווח בין חלונות
    frame_size = int(frame_duration * sample_rate)  # מספר דגימות לחלון
    step_size = int(frame_size * (1 - overlap))  # מספר דגימות להסטה בין חלונות

    # יצירת חלונות חופפים
    num_frames = (len(signal) - frame_size) // step_size + 1
    frames = [signal[i * step_size:i * step_size + frame_size] for i in range(num_frames)]

    # חישוב אנרגיה לכל חלון
    energies = [np.sum(frame ** 2) for frame in frames]

    # סיווג אנרגיה לפי פעילות קולית
    thresholds = {
        'low': 0.01,  # סף לפעילות נמוכה
        'medium': 0.05,  # סף לפעילות בינונית
    }
    classifications = []
    for energy in energies:
        if energy < thresholds['low']:
            classifications.append('low')  # אדום
        elif energy < thresholds['medium']:
            classifications.append('medium')  # כתום
        else:
            classifications.append('high')  # ירוק

    # ציר הזמן
    t = np.arange(len(signal)) / sample_rate  # ציר הזמן עבור האות

    # יצירת הגרף הראשון - האות עם חלונות
    fig1, ax1 = plt.subplots(figsize=(12, 6))
    ax1.plot(t, signal, color='blue', label='Original Signal')
    for i in range(num_frames):
        start_idx = i * step_size
        end_idx = start_idx + frame_size
        rect = plt.Rectangle(
            (t[start_idx], -1),  # פינת ההתחלה
            t[end_idx] - t[start_idx],  # רוחב
            2,  # גובה (כדי לכסות את כל האמפליטודה)
            color='red' if classifications[i] == 'low' else
            'orange' if classifications[i] == 'medium' else
            'green',
            alpha=0.2  # שקיפות
        )
        ax1.add_patch(rect)
    ax1.set_title('Signal with Overlapping Frames', fontsize=14)
    ax1.set_xlabel('Time (s)', fontsize=12)
    ax1.set_ylabel('Amplitude', fontsize=12)
    ax1.legend()
    plt.grid(True)
    plt.tight_layout()

    # שמירת הגרף הראשון
    import io
    import base64
    img1 = io.BytesIO()
    plt.savefig(img1, format='png')
    img1.seek(0)
    graph1_url = base64.b64encode(img1.getvalue()).decode()
    plt.close()

    # יצירת הגרף השני - אנרגיה של חלונות
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    ax2.bar(range(num_frames), energies, color=['red' if c == 'low' else 'orange' if c == 'medium' else 'green' for c in classifications])
    ax2.set_title('Frame Energies', fontsize=14)
    ax2.set_xlabel('Frame Number', fontsize=12)
    ax2.set_ylabel('Energy', fontsize=12)
    plt.grid(True)
    plt.tight_layout()

    # שמירת הגרף השני
    img2 = io.BytesIO()
    plt.savefig(img2, format='png')
    img2.seek(0)
    graph2_url = base64.b64encode(img2.getvalue()).decode()
    plt.close()

    # שליחת הנתונים לתבנית HTML
    return render_template('new_page.html', graph1_url=graph1_url, graph2_url=graph2_url, frame_data=zip(range(num_frames), energies, classifications))




@app.route('/vad_analysis')
def vad_analysis():
    # קריאת קובץ האודיו
    sample_rate, signal = wavfile.read('C:/Users/yaron/OneDrive/Desktop/Projects/DSP/Python/wave_file (5).wav')
    signal = signal / np.max(np.abs(signal))  # נרמול האות לטווח [-1, 1]

    # פרמטרים לחלונות
    frame_duration = 0.02  # משך חלון בשניות
    overlap = 0.25  # חפיפה של 25%
    frame_size = int(frame_duration * sample_rate)
    step_size = int(frame_size * (1 - overlap))
    num_frames = (len(signal) - frame_size) // step_size + 1

    # חלונות חופפים
    frames = [
        signal[i:i + frame_size]
        for i in range(0, len(signal) - frame_size + 1, step_size)
    ]

    # חישוב אנרגיה לחלונות
    energies = [np.sum(frame ** 2) for frame in frames]

    # סיווג חלונות
    vad_threshold = 0.01*117
    difficult_threshold = 0.5
    vad_binary = [1 if energy > vad_threshold or energy < 0.01 else 0 for energy in energies]
    hard_to_classify = [1 if vad_threshold < energy < difficult_threshold else 0 for energy in energies]

    # יצירת ציר זמן עבור האות הבינארי
    vad_signal = np.zeros(len(signal))
    for i, frame_start in enumerate(range(0, len(signal) - frame_size + 1, step_size)):
        vad_signal[frame_start:frame_start + frame_size] = vad_binary[i]

    # גרף: השוואה בין האות המקורי לאות הבינארי
    plt.figure(figsize=(12, 6))
    t = np.linspace(0, len(signal) / sample_rate, len(signal))
    plt.plot(t, signal, label='Original Signal', color='blue')
    plt.plot(t, vad_signal, label='VAD Signal', color='red', alpha=0.7)

    # הוספת צבעים לפי רמות סיווג
    for i, frame_start in enumerate(range(0, len(signal) - frame_size + 1, step_size)):
        if hard_to_classify[i]:  # חלונות קשים לסיווג בצבע סגול
            plt.axvspan(frame_start / sample_rate, (frame_start + frame_size) / sample_rate, color='purple', alpha=0.3)
        elif vad_binary[i]:  # חלונות קלים לסיווג בצבע ירוק
            plt.axvspan(frame_start / sample_rate, (frame_start + frame_size) / sample_rate, color='green', alpha=0.3)

    plt.title('VAD vs Original Signal with Classification Difficulty')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True)

    # שמירת הגרף
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    graph_url = base64.b64encode(img.getvalue()).decode()
    # נתונים לטבלה
    classifications = [{"Frame Number": i + 1, "Energy": energies[i], "Classification": "Hard" if vad_binary[i] == 0 else "Easy"}
                       for i in range(num_frames)]
    hard_count = sum(1 for vad in vad_binary if vad == 0)
    easy_count = sum(1 for vad in vad_binary if vad == 1)

    # שליחת מידע ל-HTML
    return render_template(
        'vad_analysis.html',
        graph_url=graph_url,
        classifications=classifications,
        hard_count=hard_count,
        easy_count=easy_count
    )





@app.route('/kmeans_analysis')
def kmeans_analysis():
    # קריאת קובץ האודיו
    sample_rate, signal = wavfile.read('C:/Users/yaron/OneDrive/Desktop/Projects/DSP/Python/wave_file (5).wav')
    signal = signal / np.max(np.abs(signal))  # נרמול האות לטווח [-1, 1]

    # פרמטרים לחלונות
    frame_duration = 0.02  # משך חלון בשניות
    overlap = 0.25  # חפיפה של 25%
    frame_size = int(frame_duration * sample_rate)
    step_size = int(frame_size * (1 - overlap))
    num_frames = (len(signal) - frame_size) // step_size + 1

    # חלונות חופפים
    frames = [signal[i:i + frame_size] for i in range(0, len(signal) - frame_size + 1, step_size)]

    # חישוב אנרגיה
    energies = np.array([np.sum(frame ** 2) for frame in frames]).reshape(-1, 1)

    # ביצוע KMeans
    kmeans = KMeans(n_clusters=3, random_state=42)
    clusters = kmeans.fit_predict(energies)

    # חישוב מספר הפריימים והממוצע בכל אשכול
    cluster_summary = {}
    for cluster_label in range(3):
        cluster_energies = energies[clusters == cluster_label]
        cluster_summary[cluster_label] = {
            "count": len(cluster_energies),
            "mean_energy": np.mean(cluster_energies)
        }

    # הגדרת צבעים לפי אשכולות
    cluster_colors = {0: 'orange', 1: 'yellow', 2: 'green'}
    
    # יצירת גרף
    plt.figure(figsize=(12, 6))
    t = np.linspace(0, len(signal) / sample_rate, len(signal))
    plt.plot(t, signal, label='Original Signal', color='blue')

    for i, frame_start in enumerate(range(0, len(signal) - frame_size + 1, step_size)):
        color = cluster_colors[clusters[i]]
        plt.axvspan(frame_start / sample_rate, (frame_start + frame_size) / sample_rate, color=color, alpha=0.3)

    plt.title('KMeans Analysis with Energy Clustering')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True)

    # שמירת גרף
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    graph_url = base64.b64encode(img.getvalue()).decode()

    # טבלה עם תוצאות
    classifications = [{"Frame Number": i + 1, "Energy": energies[i][0], "Cluster": clusters[i]}
                       for i in range(num_frames)]

    # שליחה ל-HTML
    return render_template(
        'kmeans_analysis.html',
        graph_url=graph_url,
        classifications=classifications,
        cluster_summary=cluster_summary
    )

    

@app.route('/base_frequency')
def base_frequency():
    # קריאת קובץ האודיו
    sample_rate, signal = wavfile.read('C:/Users/yaron/OneDrive/Desktop/Projects/DSP/Python/wave_file (6).wav')

    # נרמול האות לטווח [-1, 1]
    signal = signal / np.max(np.abs(signal))

    # חישוב ספקטרום התדר
    spectrum = np.fft.fft(signal)
    freqs = np.fft.fftfreq(len(signal), 1 / sample_rate)

    # מציאת תדר הבסיס (ההרמוניה הראשונה)
    positive_freqs = freqs[:len(freqs) // 2]
    positive_spectrum = np.abs(spectrum[:len(spectrum) // 2])
    base_frequency = positive_freqs[np.argmax(positive_spectrum)]
    base_amplitude = np.max(positive_spectrum)

    # חישוב ההרמוניות
    harmonics = []
    for i in range(1, 11):  # עד 10 הרמוניות
        harmonic_freq = base_frequency * i
        idx = (np.abs(positive_freqs - harmonic_freq)).argmin()
        harmonic_amp = positive_spectrum[idx]
        harmonics.append({'number': i, 'frequency': harmonic_freq, 'amplitude': harmonic_amp})

    # בדיקה האם תדר הבסיס הוא עם משרעת
    is_dominant = all(base_amplitude > harmonic['amplitude'] for harmonic in harmonics[1:])
    dominant_message = "Base frequency is the dominant harmonic" if is_dominant else "Base frequency is NOT the dominant harmonic"

    # יצירת גרף
    plt.figure(figsize=(10, 6))
    plt.plot(positive_freqs, positive_spectrum, label='Spectrum', color='blue')
    plt.axvline(base_frequency, color='r', linestyle='--', label=f'Base Frequency: {base_frequency:.2f} Hz')
    for harmonic in harmonics:
        plt.axvline(harmonic['frequency'], color='g', linestyle='--', label=f"Harmonic {harmonic['number']}: {harmonic['frequency']:.2f} Hz")
    plt.title('Frequency Spectrum with Harmonics')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.legend()

    # שמירת גרף כתמונה
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    graph_url = base64.b64encode(img.getvalue()).decode()
    plt.close()

    # החזרת תוצאה ל-HTML
    return render_template('base_frequency.html', graph_url=graph_url, base_frequency=base_frequency, base_amplitude=base_amplitude, dominant_message=dominant_message, harmonics=harmonics)

@app.route('/stft')
def stft_view():
    # קריאת קובץ האודיו
    sample_rate, signal = wavfile.read('C:/Users/yaron/OneDrive/Desktop/Projects/DSP/Python/wave_file (6).wav')

    # נרמול האות לטווח [-1, 1]
    signal = signal / np.max(np.abs(signal))

    # הגדרת פרמטרים של STFT
    window_length = 1024
    overlap = window_length // 2
    f, t, Zxx = stft(signal, fs=sample_rate, nperseg=window_length, noverlap=overlap)

    # חישוב המשרעת ותדרי הבסיס
    magnitude = np.abs(Zxx)
    base_freq_indices = np.argmax(magnitude, axis=0)
    base_frequencies = f[base_freq_indices]

    # הסרת תדרים כפולים
    unique_base_frequencies = np.unique(base_frequencies)
    unique_base_frequencies = unique_base_frequencies[unique_base_frequencies < 1000]  # רק תדרים מתחת ל-1000 הרץ

    # הכנת נתונים לטבלה
    table_data = pd.DataFrame({'Base Frequencies (Hz)': unique_base_frequencies})
    base_frequencies_count = len(unique_base_frequencies)

    # יצירת גרפים
    plt.figure(figsize=(12, 8))

    # ספקטרוגרם STFT מנורמל
    plt.subplot(2, 1, 1)
    plt.pcolormesh(t, f, 20 * np.log10(magnitude / np.max(magnitude)), shading='gouraud', cmap='viridis')
    plt.colorbar(label='Magnitude (dB)')
    plt.title('STFT Spectrogram (Normalized)')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')

    # גרף תדרי בסיס רגיל והפוך
    plt.subplot(2, 1,2)
    plt.plot(t, base_frequencies, 'r', label='Base Frequency', linewidth=1.5)
    plt.title('Base Frequencies Over Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.ylim(0, 1000)  # מגבלה על ציר ה-Y עד 1000 הרץ
    plt.grid(True)
    plt.legend()

    # שמירת הגרפים כתמונה
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    graph_url = base64.b64encode(img.getvalue()).decode()
    plt.close()

    # המרת הטבלה ל-HTML
    table_html = table_data.to_html(classes='table table-striped', index=False)

    return render_template('stft.html', graph_url=graph_url, sample_rate=sample_rate, base_frequencies_count=base_frequencies_count, table_html=table_html)








@app.route('/pitch_vad')
def pitch_vad_view():
     # קריאת קובץ האודיו
    sample_rate, signal = wavfile.read('C:/Users/yaron/OneDrive/Desktop/Projects/DSP/Python/wave_file (6).wav')

    

    
    # נרמול האות לטווח [-1, 1]
    signal = signal / np.max(np.abs(signal))

    # הגדרות STFT
    window_length = 1024
    overlap = window_length // 2
    f, t, Zxx = stft(signal, fs=sample_rate, nperseg=window_length, noverlap=overlap)

    # חישוב המשרעת ותדר הבסיס
    magnitude = np.abs(Zxx)
    base_freq_indices = np.argmax(magnitude, axis=0)
    base_frequencies = f[base_freq_indices]

    # זיהוי קול (VAD)
    energy_threshold = 0.1 * np.max(magnitude)
    vad_binary = np.any(magnitude > energy_threshold, axis=0)

    # זיהוי חלונות קלים וקשים לזיהוי
    hard_to_classify = (magnitude.max(axis=0) < energy_threshold / 2)

    # יצירת גרף ספקטרוגרם ותדרי בסיס
    plt.figure(figsize=(12, 8))

    # ספקטרוגרם
    plt.subplot(2, 1, 1)
    plt.pcolormesh(t, f, 20 * np.log10(magnitude), shading='gouraud', cmap='viridis')
    plt.colorbar(label='Magnitude (dB)')
    plt.title('STFT Spectrogram')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')

    # גרף תדרי בסיס עם VAD
    plt.subplot(2, 1, 2)
    plt.plot(t, base_frequencies, 'r', label='Base Frequency (Pitch)', linewidth=1.5)
    plt.fill_between(t, 0, base_frequencies, where=vad_binary, color='green', alpha=0.3, label='Voiced (VAD)')
    plt.fill_between(t, 0, base_frequencies, where=hard_to_classify, color='purple', alpha=0.3, label='Hard to Classify')
    plt.title('Pitch Contour with VAD')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.legend()
    plt.grid(True)

    # שמירת הגרף
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    graph_url = base64.b64encode(img.getvalue()).decode()
    plt.close()

    return render_template('pitch_vad.html', graph_url=graph_url, sample_rate=sample_rate)




@app.route('/signal_synthesis')
def signal_synthesis_view():
    # קריאת קובץ האודיו
    sample_rate, signal = wavfile.read('C:/Users/yaron/OneDrive/Desktop/Projects/DSP/Python/wave_file (6).wav')
    
    # נרמול האות לטווח [-1, 1]
    signal = signal / np.max(np.abs(signal))
    
    # STFT הגדרת פרמטרים
    window_length = 1024
    overlap = window_length // 2
    f, t, Zxx = stft(signal, fs=sample_rate, nperseg=window_length, noverlap=overlap)
    
    # סינון תדרים בעלי משרעת נמוכה
    magnitude = np.abs(Zxx)
    threshold = np.percentile(magnitude, 20)  # סף משרעת
    harmonic_mask = magnitude >= threshold
    Zxx_filtered = Zxx * harmonic_mask

    # IFFT סינתזה מחדש
    _, synthesized_signal = istft(Zxx_filtered, fs=sample_rate, nperseg=window_length, noverlap=overlap)

    # תיאום אורך האות
    min_length = min(len(t), len(synthesized_signal))
    t = t[:min_length]
    synthesized_signal = synthesized_signal[:min_length]

    # שמירת האודיו כקובץ WAV
    write('static/synthesized_signal.wav', sample_rate, synthesized_signal.astype(np.float32))

    # יצירת הגרפים
    plt.figure(figsize=(12, 8))
    
    # ספקטרוגרם STFT
    plt.subplot(2, 1, 1)
    plt.pcolormesh(t, f, 20 * np.log10(magnitude), shading='gouraud', cmap='viridis')
    plt.colorbar(label='Magnitude (dB)')
    plt.title('STFT Spectrogram')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    
    # גרף סינתזה
    plt.subplot(2, 1, 2)
    plt.plot(t, synthesized_signal, label='Synthesized Signal', color='r', alpha=0.7)
    plt.title('Synthesized Signal from Harmonics')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True)
    
    # שמירת הגרף כתמונה
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    graph_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    
    return render_template('signal_synthesis.html', graph_url=graph_url, sample_rate=sample_rate)

if __name__ == '__main__':
    app.run(debug=True)
