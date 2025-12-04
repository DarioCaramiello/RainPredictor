import argparse
import rasterio
import matplotlib.pyplot as plt

def main():
    """Plot input vs predicted radar frame side-by-side in physical coordinates."""
    parser = argparse.ArgumentParser(
        description="Compare an input GeoTIFF frame vs a predicted frame in lon/lat."
    )
    parser.add_argument("--input", type=str, required=True, help="Path to original input GeoTIFF.")
    parser.add_argument("--pred", type=str, required=True, help="Path to predicted GeoTIFF.")
    parser.add_argument("--title", type=str, default="Input vs Prediction", help="Figure title.")
    args = parser.parse_args()

    with rasterio.open(args.input) as src_in:
        img_in = src_in.read(1)
        transform = src_in.transform
        crs = src_in.crs
        width = src_in.width
        height = src_in.height

    with rasterio.open(args.pred) as src_pr:
        img_pr = src_pr.read(1)

    left = transform.c
    top = transform.f
    res_x = transform.a
    res_y = transform.e
    right = left + width * res_x
    bottom = top + height * res_y
    extent = [left, right, bottom, top]

    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharex=True, sharey=True)
    im0 = axes[0].imshow(img_in, extent=extent, origin="upper")
    axes[0].set_title("Input")
    axes[0].set_xlabel("Longitude")
    axes[0].set_ylabel("Latitude")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(img_pr, extent=extent, origin="upper")
    axes[1].set_title("Prediction")
    axes[1].set_xlabel("Longitude")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    fig.suptitle(args.title)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
