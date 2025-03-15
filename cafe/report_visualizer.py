import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Optional, List, Any, Union

class ReportVisualizer:
    def __init__(self, figsize_default=(12, 8)):
        self.figsize_default = figsize_default
        self.color_palette = {
            'primary': '#1a53ff',      # Azul principal
            'secondary': '#ff6600',    # Laranja
            'success': '#28a745',      # Verde
            'danger': '#dc3545',       # Vermelho
            'warning': '#ffc107',      # Amarelo
            'info': '#17a2b8',         # Ciano
            'light': '#f8f9fa',        # Cinza claro
            'dark': '#343a40'          # Cinza escuro
        }
        
        self.importance_colors = {
            'Muito Alta': '#1a53ff',   # Azul escuro
            'Alta': '#4d79ff',         # Azul médio
            'Média': '#80a0ff',        # Azul claro
            'Baixa': '#b3c6ff',        # Azul muito claro
            'Muito Baixa': '#e6ecff'   # Azul quase branco
        }
    
    def visualize_missing_values(self, missing_report: pd.DataFrame, figsize=None, top_n=20):
        if missing_report.empty:
            print("Não há valores ausentes para visualizar.")
            return None
        
        # Limitar ao top_n colunas com mais valores ausentes
        missing_report = missing_report.head(top_n)
        
        # Criar figura
        fig, ax = plt.subplots(figsize=figsize or self.figsize_default)
        
        # Gráfico de barras para valores ausentes
        bars = ax.barh(missing_report['coluna'], missing_report['porcentagem'], color=self.color_palette['info'])
        
        # Adicionar valores nas barras
        for bar in bars:
            width = bar.get_width()
            ax.text(width + 1, bar.get_y() + bar.get_height()/2, f'{width:.1f}%', ha='left', va='center')
        
        ax.set_xlabel('Porcentagem de Valores Ausentes (%)')
        ax.set_ylabel('Coluna')
        ax.set_title('Valores Ausentes por Coluna')
        ax.grid(axis='x', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        
        return fig
    
    def visualize_outliers(self, outliers_report: pd.DataFrame, df: pd.DataFrame,  columns=None, figsize=None, max_cols=5):
        if outliers_report.empty:
            print("Não foram detectados outliers significativos para visualizar.")
            return None
        
        if columns is not None:
            cols_to_plot = [col for col in columns if col in outliers_report['coluna'].values]
            if not cols_to_plot:
                print("Nenhuma das colunas especificadas tem outliers significativos.")
                return None
        else:
            cols_to_plot = outliers_report.head(max_cols)['coluna'].tolist()
        
        n_cols = len(cols_to_plot)
        
        if n_cols == 0:
            print("Não há colunas com outliers para visualizar.")
            return None
        
        fig_height = 4 * n_cols  # Altura proporcional ao número de colunas
        this_figsize = figsize or (self.figsize_default[0], fig_height)
        
        fig, axes = plt.subplots(n_cols, 1, figsize=this_figsize)
        if n_cols == 1:
            axes = [axes]
        
        for i, col in enumerate(cols_to_plot):
            col_data = df[col].dropna()
            row = outliers_report[outliers_report['coluna'] == col].iloc[0]
            lower_bound = row['limite_inferior']
            upper_bound = row['limite_superior']
            
            # Criar boxplot
            ax = axes[i]
            sns.boxplot(x=col_data, ax=ax, color=self.color_palette['light'])
            
            # Adicionar marcação dos limites
            ax.axvline(x=lower_bound, color=self.color_palette['danger'], linestyle='--', alpha=0.7)
            ax.axvline(x=upper_bound, color=self.color_palette['danger'], linestyle='--', alpha=0.7)
            
            # Adicionar título e informações
            ax.set_title(f"{col}: {row['num_outliers']} outliers ({row['porcentagem']}%)")
            ax.text(0.02, 0.85, f"Limite inferior: {lower_bound:.2f}", transform=ax.transAxes)
            ax.text(0.02, 0.75, f"Limite superior: {upper_bound:.2f}", transform=ax.transAxes)
        
        plt.tight_layout()
        
        return fig
    
    def visualize_feature_importance(self, importance_report: pd.DataFrame, figsize=None, top_n=15):
        if importance_report.empty:
            print("Não há dados de importância de features para visualizar.")
            return None
        
        importance_report = importance_report.head(top_n)
        
        fig, ax = plt.subplots(figsize=figsize or self.figsize_default)
        
        colors = [self.importance_colors.get(cat, '#808080') for cat in importance_report['categoria']]
        
        bars = ax.barh(importance_report['feature'], importance_report['normalized_importance'], color=colors)
        
        for bar in bars:
            width = bar.get_width()
            ax.text(width + 0.5, bar.get_y() + bar.get_height()/2, f'{width:.1f}%', ha='left', va='center')
        
        ax.set_xlabel('Importância Normalizada (%)')
        ax.set_ylabel('Feature')
        ax.set_title('Importância das Features para a Variável Alvo')
        ax.grid(axis='x', linestyle='--', alpha=0.7)
        
        # Adicionar legenda para categorias
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=color, label=cat) for cat, color in self.importance_colors.items()]
        ax.legend(handles=legend_elements, title='Categoria de Importância', loc='lower right')
        
        plt.tight_layout()
        
        return fig
    
    def visualize_transformations(self, validation_results: Dict, transformation_stats: Dict = None, figsize=None):
        
        if not validation_results:
            print("Não há resultados de validação para visualizar.")
            return None
        
        # Criar figura
        fig, axes = plt.subplots(2, 2, figsize=figsize or (14, 10))
        
        # 1. Gráfico de comparação de performance
        ax1 = axes[0, 0]
        performance = [
            validation_results['performance_original'],
            validation_results['performance_transformed']
        ]
        
        # Definir cores baseadas no ganho de performance
        if validation_results['performance_diff'] >= 0:
            colors = [self.color_palette['primary'], self.color_palette['success']]
        else:
            colors = [self.color_palette['primary'], self.color_palette['danger']]
        
        bars = ax1.bar(['Original', 'Transformado'], performance, color=colors)
        
        # Adicionar valores nas barras
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2, height + 0.01,
                     f'{height:.4f}', ha='center', va='bottom')
        
        ax1.set_title('Comparação de Performance')
        ax1.set_ylabel('Performance (Métrica)')
        ax1.grid(axis='y', linestyle='--', alpha=0.7)
        
        # 2. Comparação de número de features
        ax2 = axes[0, 1]
        
        if transformation_stats and transformation_stats['dimensoes_originais'] is not None:
            original_features = transformation_stats['dimensoes_originais'][1]
            transformed_features = transformation_stats['dimensoes_transformadas'][1]
            feature_reduction = transformation_stats['reducao_features_pct']
            
            # Definir cores baseadas na redução de features
            if feature_reduction >= 0:
                feature_colors = [self.color_palette['primary'], self.color_palette['success']]
                title_suffix = f"Redução de {feature_reduction:.1f}%"
            else:
                feature_colors = [self.color_palette['primary'], self.color_palette['warning']]
                title_suffix = f"Aumento de {abs(feature_reduction):.1f}%"
                
            # Criar gráfico de barras para número de features antes/depois
            feature_bars = ax2.bar(['Original', 'Transformado'], [original_features, transformed_features], color=feature_colors)
                
            ax2.set_title(f'Comparação de Features - {title_suffix}')
            ax2.set_ylabel('Número de Features')
            ax2.grid(axis='y', linestyle='--', alpha=0.7)
            
            # Adicionar valores nas barras
            for bar in feature_bars:
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2, height + 0.5,f"{int(height)}", ha='center', va='bottom')
        else:
            ax2.text(0.5, 0.5, "Informações de features não disponíveis", ha='center', va='center', transform=ax2.transAxes)
        
        # 3. Performance por fold
        ax3 = axes[1, 0]
        if 'scores_original' in validation_results and 'scores_transformed' in validation_results:
            folds = list(range(1, len(validation_results['scores_original'])+1))
            
            ax3.plot(folds, validation_results['scores_original'], 'o-', 
                    label='Original', color=self.color_palette['primary'])
            
            # Definir cor baseada no ganho de performance
            transform_color = self.color_palette['success'] if validation_results['performance_diff'] >= 0 else self.color_palette['danger']
            
            ax3.plot(folds, validation_results['scores_transformed'], 'o-', 
                    label='Transformado', color=transform_color)
            
            ax3.set_title('Performance por Fold de Validação Cruzada')
            ax3.set_xlabel('Fold')
            ax3.set_ylabel('Performance')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        else:
            ax3.text(0.5, 0.5, "Scores por fold não disponíveis", ha='center', va='center', transform=ax3.transAxes)
        
        # 4. Texto com resumo e decisão
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        performance_diff = validation_results['performance_diff']
        performance_diff_pct = validation_results['performance_diff_pct']
        best_choice = validation_results['best_choice']
        
        # Obter dimensões das features (se disponíveis)
        if transformation_stats and transformation_stats['dimensoes_originais'] is not None:
            orig_features = transformation_stats['dimensoes_originais'][1]
            trans_features = transformation_stats['dimensoes_transformadas'][1]
            feature_red_pct = abs(transformation_stats['reducao_features_pct'])
        else:
            orig_features = "N/A"
            trans_features = "N/A"
            feature_red_pct = "N/A"
        
        text = f"""
        RESUMO DA VALIDAÇÃO

        Performance:
        - Original:     {validation_results['performance_original']:.4f}
        - Transformado: {validation_results['performance_transformed']:.4f}
        - Diferença:    {performance_diff:.4f} ({performance_diff_pct:.2f}%)

        Features:
        - Original:     {orig_features}
        - Transformado: {trans_features}
        - Redução:      {feature_red_pct}%

        DECISÃO: Usar dados {best_choice.upper()}
        """
        
        ax4.text(0.1, 0.9, text, fontsize=12, va='top', family='monospace')
        
        plt.tight_layout()
        
        return fig
    
    def visualize_data_distribution(self, df: pd.DataFrame, columns=None, figsize=None, max_cols=6):
        if columns is not None:
            numeric_cols = [col for col in columns if col in df.columns and pd.api.types.is_numeric_dtype(df[col].dtype)]
        else:
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        
        # Limitar o número de colunas
        if len(numeric_cols) > max_cols:
            numeric_cols = numeric_cols[:max_cols]
        
        if not numeric_cols:
            print("Não há colunas numéricas para visualizar a distribuição.")
            return None
        
        # Determinar layout da figura
        n_cols = min(3, len(numeric_cols))
        n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
        
        # Criar figura
        fig_width = n_cols * 5
        fig_height = n_rows * 4
        this_figsize = figsize or (fig_width, fig_height)
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=this_figsize)
        if n_rows * n_cols == 1:
            axes = np.array([axes])
        axes = axes.flatten()
        
        # Para cada coluna, criar histograma e kde
        for i, col in enumerate(numeric_cols):
            if i < len(axes):
                ax = axes[i]
                sns.histplot(df[col].dropna(), kde=True, ax=ax, color=self.color_palette['primary'])
                ax.set_title(f'Distribuição de {col}')
                ax.set_xlabel(col)
                ax.set_ylabel('Frequência')
        
        # Ocultar subplots vazios
        for i in range(len(numeric_cols), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        return fig
    
    def visualize_correlation_matrix(self, df: pd.DataFrame, figsize=None, target_col=None):
        numeric_df = df.select_dtypes(include=['number'])
        
        if numeric_df.empty:
            print("Não há colunas numéricas para calcular correlações.")
            return None
        
        # Calcular matriz de correlação
        corr_matrix = numeric_df.corr()
        
        # Criar figura
        fig, ax = plt.subplots(figsize=figsize or (12, 10))
        
        # Criar máscara para triângulo superior
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        # Criar heatmap
        sns.heatmap(corr_matrix, mask=mask, cmap='coolwarm', annot=False, center=0, square=True, linewidths=.5, ax=ax)
        
        ax.set_title('Matriz de Correlação')
        
        # Se houver coluna alvo, criar gráfico adicional mostrando correlações com o target
        if target_col and target_col in corr_matrix.columns:
            # Criar figura adicional
            fig2, ax2 = plt.subplots(figsize=(10, 6))
            
            # Ordenar correlações com o target
            target_corr = corr_matrix[target_col].drop(target_col).sort_values(ascending=False)
            
            # Definir cores baseadas no valor da correlação
            colors = [self.color_palette['success'] if x >= 0 else self.color_palette['danger'] 
                     for x in target_corr]
            
            # Criar gráfico de barras
            bars = ax2.barh(target_corr.index, target_corr, color=colors)
            
            # Adicionar valores nas barras
            for bar in bars:
                width = bar.get_width()
                ax2.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                        f'{width:.2f}', va='center')
            
            ax2.set_title(f'Correlação com {target_col}')
            ax2.set_xlabel('Coeficiente de Correlação')
            ax2.axvline(x=0, color='black', linestyle='-', alpha=0.3)
            ax2.grid(axis='x', linestyle='--', alpha=0.7)
            
            return fig, fig2
        
        return fig