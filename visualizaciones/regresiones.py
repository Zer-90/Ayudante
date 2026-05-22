import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def par_real_predicho(y_test, y_pred, magnitud=""):
    # 1. Crear la cuadrícula conjunta para tener control total
    g = sns.JointGrid(x=y_test, y=y_pred, marginal_ticks=True)

    # 2. Gráfico de dispersión en el área central
    g.plot_joint(sns.scatterplot, color="steelblue", alpha=0.7, edgecolor="white", s=60)

    # 3. Línea bisectriz fija (y = x) desde min(y_test) a max(y_test)
    min_v, max_v = np.min(y_test), np.max(y_test)
    g.ax_joint.plot([min_v, max_v], [min_v, max_v],
                    color='red', linestyle='--', linewidth=2, label='Bisectriz (y=x)')

    # IMPORTANTE: Forzar relación de aspecto 1:1 para que la bisectriz sea visualmente diagonal real
    g.ax_joint.set_aspect('equal', adjustable='box')

    # 4. KDE en los márgenes (reemplaza los histogramas por defecto)
    g.plot_marginals(sns.kdeplot, fill=True, color="steelblue", alpha=0.4)

    # (OPCIONAL) Si prefieres que el eje Y marginal muestre explícitamente la distribución
    # de los RESIDUOS en lugar de la distribución de y_pred, descomenta estas líneas:
    # residuos = y_pred - y_test
    # g.ax_marg_y.clear()
    # sns.kdeplot(residuos, ax=g.ax_marg_y, fill=True, color="darkorange", alpha=0.5)

    # 5. Etiquetas y formato final
    g.ax_joint.set_xlabel(f'Valor real de {magnitud}')
    g.ax_joint.set_ylabel(f'Valor predicho de {magnitud}')
    g.ax_joint.legend()

    plt.tight_layout()
    plt.show()

    if __name__ == "__main__":
        import pandas as pd
        y_test = pd.Series([1,2,3])
        y_pred = pd.Series([0.95,1.87,2.64])
        par_real_predicho(y_test, y_pred, "Valores")
        plt.savefig('regresiones.png')