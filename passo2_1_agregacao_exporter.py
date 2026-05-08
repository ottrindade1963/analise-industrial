"""Exportação dos datasets agregados em CSV e XLSX."""

import pandas as pd
from passo2_1_agregacao_config import OUTPUT_DIR_M1, OUTPUT_DIR_M2, OUTPUT_DIR_M3


def exportar_dataset(df, output_dir, nome_base, metodo_descricao):
    """Exporta um dataset agregado em CSV e XLSX com metadados."""
    
    csv_path = f"{output_dir}/{nome_base}.csv"
    xlsx_path = f"{output_dir}/{nome_base}.xlsx"
    
    # CSV
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    
    # XLSX com aba de metadados
    with pd.ExcelWriter(xlsx_path, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Dados', index=False)
        
        # Metadados
        meta = pd.DataFrame({
            'Campo': ['Método', 'Registros', 'Países', 'Período',
                      'Colunas', 'Gerado em', 'Fonte Quantitativa', 'Fonte Qualitativa'],
            'Valor': [
                metodo_descricao,
                str(len(df)),
                str(df['country_code'].nunique()),
                f"{df['year'].min()} - {df['year'].max()}",
                str(len(df.columns)),
                pd.Timestamp.now().strftime('%Y-%m-%d %H:%M'),
                'WDI - World Development Indicators (Banco Mundial)',
                'WGI + ICRG/QoG (Banco Mundial + Univ. Gothenburg)'
            ]
        })
        meta.to_excel(writer, sheet_name='Metadados', index=False)
    
    print(f"    📄 {csv_path}")
    print(f"    📊 {xlsx_path}")


def exportar_todos(df_m1, df_m2, df_m3):
    """Exporta os 3 datasets agregados (6 ficheiros no total)."""
    
    print("\n  ── Exportando Datasets ──")
    
    print("\n  Método 1 (Inner Join):")
    exportar_dataset(df_m1, OUTPUT_DIR_M1, "agregado_inner",
                     "Inner Join — Junção Interna Estrita")
    
    print("\n  Método 2 (Left Join + Imputação):")
    exportar_dataset(df_m2, OUTPUT_DIR_M2, "agregado_left_imputado",
                     "Left Join com Imputação Temporal")
    
    print("\n  Método 3 (Outer Join + Fonte):")
    exportar_dataset(df_m3, OUTPUT_DIR_M3, "agregado_outer_completo",
                     "Outer Join com Indicador de Fonte")
