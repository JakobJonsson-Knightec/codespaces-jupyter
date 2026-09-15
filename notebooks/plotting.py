from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import tree

def create_plots(clf, feature_names):
    """Save educational plots for the trained tree and its test results."""
    plot_dir = Path("plots")
    plot_dir.mkdir(exist_ok=True)

    plt.figure(figsize=(14, 8))
    tree.plot_tree(
        clf,
        feature_names=feature_names,
        class_names=clf.classes_,
        impurity=True,
        filled=True,
        rounded=True,
        fontsize=9,
    )
    plt.title("Beslutsträdet: vilka frågor ställer trädet?")
    plt.tight_layout()
    plt.savefig(plot_dir / "beslutstrad.png", dpi=150)
    plt.show()

    importance = pd.Series(clf.feature_importances_, index=feature_names)
    importance = importance.sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    importance.plot.barh(ax=ax, color="#2a9d8f")
    ax.set_title("Vilka egenskaper använder trädet mest?")
    ax.set_xlabel("Viktighet")
    ax.set_ylabel("Egenskap")
    fig.tight_layout()
    fig.savefig(plot_dir / "feature_viktighet.png", dpi=150)
    plt.show()

    # print("Sparade bilder i plots/: beslutstrad.png, feature_viktighet.png")
