import csv
from pathlib import Path
import matplotlib.pyplot as plt


def summary_csv_to_png(csv_path: Path, out_png: Path, title: str) -> None:
    with open(csv_path, "r") as f:
        reader = csv.reader(f)
        rows = list(reader)

    header = rows[0]
    data = rows[1:]

    fig, ax = plt.subplots(figsize=(10, 2 + 0.4 * len(data)))
    ax.axis("off")

    table = ax.table(
        cellText=data,
        colLabels=header,
        loc="center",
        cellLoc="center",
    )

    # Make header bold
    for (row, col), cell in table.get_celld().items():
        if row == 0:  # header row
            cell.set_text_props(weight="bold")

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.4)

    ax.set_title(title, pad=20)

    plt.tight_layout()
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    dataset_name = "abz9"
    results_dir = Path("results") / dataset_name

    csv_path = results_dir / f"summary_{dataset_name}.csv"
    out_png = results_dir / f"summary_{dataset_name}.png"

    summary_csv_to_png(
        csv_path,
        out_png,
        title=f"Summary of GA Results for {dataset_name.upper()} Dataset",
    )

    print(f"Table saved to {out_png}")
