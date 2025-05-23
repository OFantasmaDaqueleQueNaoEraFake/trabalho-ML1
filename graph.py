import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


def grafic(metrics, metrics_nome):

        df_metrics = pd.DataFrame(list(metrics.items()), columns=["Metrics", "Value"]);

        barplot = sns.barplot(x="Value", y="Metrics", data=df_metrics, palette='coolwarm');

        for index, value in enumerate(df_metrics["Value"]):
            plt.text(value + 0.01, index, f'{value:.4f}', va='center');

        plt.title(metrics_nome);
        plt.grid(True, axis='x');
        plt.xlabel("Score");
        plt.xlim(0, 1);
        plt.tight_layout();
        plt.show();

def heatmappers(data):
      
      import numpy as np
      data_normalized = np.round(data/np.sum(data, axis=1).reshape(-1,1), 2);
      aoba = sns.cubehelix_palette(as_cmap=True)
      sns.heatmap(data_normalized, cmap=aoba, annot=True);
      plt.xlabel("Predicted");
      plt.ylabel("Actual");
      plt.show();