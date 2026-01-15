import streamlit as st
from PIL import Image
from filters import apply_filter_sequential, apply_filter_parallel
import matplotlib.pyplot as plt
import os

st.set_page_config(layout="wide")

st.title("Parallel Image Processing")
st.subheader("Group Members:")

st.write("Hamza Ahmed Khan")
st.write("Sibtain Ahmed")
st.write("Munesh Kumar")
st.write("Pawan Mahesh")
st.write("Ahmed Ali Khokhar")

st.write("---")

# Mode selection
mode = st.radio("Select Mode", ["Single Image", "Batch Processing"])

if not os.path.exists("images/output"):
    os.makedirs("images/output")

# ============= SINGLE IMAGE MODE =============
if mode == "Single Image":
    uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    filter_type = st.selectbox("Select a filter", ["Grayscale", "Blur", "Edge Detection"])
    num_threads = st.slider("Threads", 1, 8, 4)
    
    if uploaded:
        image = Image.open(uploaded).convert("RGB")
        width, height = image.size
        pixel_count = width * height
        
        st.image(image, caption="Original Image", width=400)

        # Sequential
        seq_img, seq_time = apply_filter_sequential(image, filter_type)
        seq_path = f"images/output/seq_{filter_type}.jpg"
        seq_img.save(seq_path)

        # Parallel
        par_img, par_time = apply_filter_parallel(image, filter_type, num_threads)
        par_path = f"images/output/par_{filter_type}.jpg"
        par_img.save(par_path)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Sequential Result")
            st.image(seq_img, width=400)
            st.write(f"Time: {seq_time:.4f} seconds")

        with col2:
            st.subheader("Parallel Result")
            st.image(par_img, width=400)
            st.write(f"Time: {par_time:.4f} seconds")

        # Performance Metrics
        st.write("---")
        st.subheader("📊 Performance Metrics")
        
        speedup = seq_time / par_time if par_time > 0 else 0
        seq_throughput = pixel_count / seq_time if seq_time > 0 else 0
        par_throughput = pixel_count / par_time if par_time > 0 else 0
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Speedup Factor", f"{speedup:.2f}x")
        with col2:
            st.metric("Seq Throughput", f"{seq_throughput/1e6:.2f}M pixels/s")
        with col3:
            st.metric("Par Throughput", f"{par_throughput/1e6:.2f}M pixels/s")
        with col4:
            efficiency = (speedup / num_threads) * 100 if num_threads > 0 else 0
            st.metric("Parallel Efficiency", f"{efficiency:.1f}%")

        # Time Comparison Chart
        st.write("---")
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Execution Time Comparison")
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.bar(["Sequential", "Parallel"], [seq_time, par_time], color=["gray", "skyblue"], width=0.5)
            ax.set_ylabel("Time (seconds)")
            ax.set_title(f"Processing Time ({filter_type})")
            plt.tight_layout()
            st.pyplot(fig)
        
        # Optimal Thread Count Analysis
        with col2:
            st.subheader("Optimal Thread Count")
            thread_counts = range(1, 9)
            parallel_times = []
            
            for threads in thread_counts:
                _, p_time = apply_filter_parallel(image, filter_type, threads)
                parallel_times.append(p_time)
            
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(thread_counts, parallel_times, marker='o', color='skyblue', linewidth=2, markersize=8)
            ax.axhline(y=seq_time, color='gray', linestyle='--', label='Sequential Time', linewidth=2)
            ax.set_xlabel("Number of Threads")
            ax.set_ylabel("Time (seconds)")
            ax.set_title(f"Performance vs Thread Count ({filter_type})")
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_xticks(thread_counts)
            plt.tight_layout()
            st.pyplot(fig)
            
            optimal_threads = thread_counts[parallel_times.index(min(parallel_times))]
            st.success(f"✓ Optimal thread count: **{optimal_threads}**")

        # Speedup Chart
        st.write("---")
        st.subheader("Speedup Factor vs Thread Count")
        speedup_factors = [seq_time / t for t in parallel_times]
        
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(thread_counts, speedup_factors, marker='o', color='green', linewidth=2.5, markersize=8, label='Actual Speedup')
        ax.plot(thread_counts, thread_counts, linestyle='--', color='red', linewidth=2, label='Ideal Speedup (Linear)')
        ax.set_xlabel("Number of Threads")
        ax.set_ylabel("Speedup Factor")
        ax.set_title(f"Speedup Analysis ({filter_type})")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xticks(thread_counts)
        plt.tight_layout()
        st.pyplot(fig)

    else:
        st.info("Upload an image to begin.")

# ============= BATCH PROCESSING MODE =============
else:
    st.subheader("Batch Image Processing")
    uploaded_files = st.file_uploader("Upload multiple images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
    filter_type = st.selectbox("Select a filter", ["Grayscale", "Blur", "Edge Detection"])
    num_threads = st.slider("Threads per Image", 1, 8, 4)
    
    if uploaded_files:
        st.write(f"Processing {len(uploaded_files)} image(s)...")
        
        batch_results = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for idx, uploaded_file in enumerate(uploaded_files):
            status_text.text(f"Processing image {idx + 1}/{len(uploaded_files)}: {uploaded_file.name}")
            
            image = Image.open(uploaded_file).convert("RGB")
            width, height = image.size
            pixel_count = width * height
            
            # Sequential processing
            seq_img, seq_time = apply_filter_sequential(image, filter_type)
            
            # Parallel processing
            par_img, par_time = apply_filter_parallel(image, filter_type, num_threads)
            
            # Save results
            seq_filename = f"images/output/batch_seq_{idx}_{filter_type}.jpg"
            par_filename = f"images/output/batch_par_{idx}_{filter_type}.jpg"
            seq_img.save(seq_filename)
            par_img.save(par_filename)
            
            speedup = seq_time / par_time if par_time > 0 else 0
            seq_throughput = pixel_count / seq_time if seq_time > 0 else 0
            par_throughput = pixel_count / par_time if par_time > 0 else 0
            
            batch_results.append({
                "Image": uploaded_file.name,
                "Seq Time (s)": f"{seq_time:.4f}",
                "Par Time (s)": f"{par_time:.4f}",
                "Speedup": f"{speedup:.2f}x",
                "Seq Throughput (M pix/s)": f"{seq_throughput/1e6:.2f}",
                "Par Throughput (M pix/s)": f"{par_throughput/1e6:.2f}"
            })
            
            progress_bar.progress((idx + 1) / len(uploaded_files))
        
        status_text.success("✓ Batch processing complete!")
        progress_bar.empty()
        
        # Display Results Table
        st.write("---")
        st.subheader("📊 Batch Processing Results")
        st.table(batch_results)
        
        # Summary Statistics
        st.write("---")
        st.subheader("Summary Statistics")
        import pandas as pd
        df = pd.DataFrame(batch_results)
        
        # Convert string values to float for calculations
        seq_times = [float(x) for x in df["Seq Time (s)"]]
        par_times = [float(x) for x in df["Par Time (s)"]]
        speedups = [float(x.replace('x', '')) for x in df["Speedup"]]
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Seq Time", f"{sum(seq_times):.2f}s")
        with col2:
            st.metric("Total Par Time", f"{sum(par_times):.2f}s")
        with col3:
            st.metric("Avg Speedup", f"{sum(speedups)/len(speedups):.2f}x")
        with col4:
            total_speedup = sum(seq_times) / sum(par_times) if sum(par_times) > 0 else 0
            st.metric("Overall Speedup", f"{total_speedup:.2f}x")
    
    else:
        st.info("Upload multiple images to begin batch processing.")
