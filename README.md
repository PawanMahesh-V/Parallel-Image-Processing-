
# Parallel Image Processing

An interactive Streamlit app for applying image processing filters (Grayscale, Blur, Edge Detection) using both sequential and parallel processing. The app demonstrates the performance benefits of parallelization and provides visual and statistical comparisons.

🌐 **Live Demo:** [parallelimageprocessing.streamlit.app](https://parallelimageprocessing.streamlit.app/)

## Features
- **Single Image Mode:**
	- Upload an image and apply a filter.
	- Compare sequential vs parallel processing (choose thread count).
	- Visualize results, speedup, throughput, and efficiency.
	- Performance charts for execution time and speedup vs thread count.
- **Batch Processing Mode:**
	- Upload multiple images for batch filtering.
	- View per-image and summary performance metrics.
- **Filters Supported:** Grayscale, Blur, Edge Detection
- **Parallelization:** Uses Python's `concurrent.futures` and NumPy for efficient processing.

## How to Run Locally
1. **Clone the repository**
2. **Install dependencies:**
	 ```bash
	 pip install -r requirements.txt
	 ```
3. **Run the app:**
	 ```bash
	 streamlit run app.py
	 ```
4. Open the provided local URL in your browser.

## Project Structure
```
Parallel_Image_Processing/
	app.py            # Streamlit app
	filters.py        # Image filter logic (sequential & parallel)
	requirements.txt  # Python dependencies
	images/output/    # Output images
```

## Authors
- Hamza Ahmed Khan (2212341)
- Sibtain Ahmed (2212271)
- Munesh Kumar (2212260)
- Pawan Mahesh (2212263)
- Ahmed Ali Khokhar (2212243)

---
For questions or feedback, please open an issue or contact the authors.
