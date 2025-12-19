using Images, ImageMagick, ImageIO

function create_gif_from_folder(folder_path::String, output_path::String; delay::Float64=0.1)
    # Collect all image file paths with supported extensions
    image_files = sort(filter(f -> any(ext -> endswith(lowercase(f), ext), [".png", ".jpg", ".jpeg"]), readdir(folder_path, join=true)))

    # Load images
    frames = [load(f) for f in image_files]

    # Save as GIF
    save(output_path, cat(frames...; dims=3), fps=1/delay)
    println("GIF saved to: $output_path")
end

# Example usage:
create_gif_from_folder("jl_mxNaEz/", "output.gif", delay=0.05)
