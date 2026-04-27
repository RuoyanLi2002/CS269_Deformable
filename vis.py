import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# --- CONFIGURATION ---
INPUT_PATH = 'valid.pkl'
OUTPUT_GIF = 'trajectory_0.gif'
FPS = 20

def create_gif():
    print(f"Loading data from {INPUT_PATH}...")
    with open(INPUT_PATH, 'rb') as f:
        data = pickle.load(f)
    
    # Extract the first trajectory
    # positions shape: (Frames, Particles, 2)
    # types shape: (Particles, 1)
    positions = data['positions'][0]
    particle_types = data['types'][0].flatten() 
    
    num_frames = positions.shape[0]
    print(f"Loaded trajectory with {num_frames} frames.")

    # --- SETUP PLOT ---
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Find global min/max across all frames to keep axes fixed
    x_min, x_max = positions[:, :, 0].min(), positions[:, :, 0].max()
    y_min, y_max = positions[:, :, 1].min(), positions[:, :, 1].max()
    
    # Add a 5% margin
    x_margin = (x_max - x_min) * 0.05
    y_margin = (y_max - y_min) * 0.05
    
    ax.set_xlim(x_min - x_margin, x_max + x_margin)
    ax.set_ylim(y_min - y_margin, y_max + y_margin)
    ax.set_aspect('equal') # Crucial for physical simulations so shapes don't distort
    ax.set_title("Trajectory 0 - Frame 0")
    
    # Initialize the scatter plot
    # Using 'tab10' colormap to distinctively color different particle types
    scat = ax.scatter(
        positions[0, :, 0], 
        positions[0, :, 1], 
        c=particle_types, 
        cmap='tab10', 
        s=10,       # Point size
        alpha=0.8   # Slight transparency
    )

    # --- ANIMATION FUNCTION ---
    def update(frame):
        # Update the positions of the scatter plot
        scat.set_offsets(positions[frame])
        ax.set_title(f"Trajectory 0 - Frame {frame}/{num_frames}")
        
        # Print progress to console
        if frame % 10 == 0:
            print(f"Rendering frame {frame}/{num_frames}...", end='\r')
            
        return scat,

    # --- GENERATE & SAVE ---
    print(f"Generating animation...")
    ani = animation.FuncAnimation(
        fig, 
        update, 
        frames=num_frames, 
        interval=1000/FPS, 
        blit=True
    )
    
    print(f"Saving to {OUTPUT_GIF} (This may take a minute)...")
    # PillowWriter is built into matplotlib and doesn't require FFmpeg
    writer = animation.PillowWriter(fps=FPS)
    ani.save(OUTPUT_GIF, writer=writer)
    
    print(f"\nDone! GIF saved to {OUTPUT_GIF}")

if __name__ == "__main__":
    create_gif()