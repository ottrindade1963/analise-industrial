"""
Orquestrador Mestre - Pipeline Completo (Passos 1 a 9)
Este script executa todo o fluxo de dados, desde a extração até às análises avançadas.
"""
import os
import sys
import time

def print_header(texto):
    print("\n" + "=" * 80)
    print(f"🚀 {texto}")
    print("=" * 80 + "\n")

def executar_passo(nome_script, descricao):
    print_header(descricao)
    inicio = time.time()
    
    # Executar o script
    resultado = os.system(f"python3 {nome_script}")
    
    fim = time.time()
    duracao = fim - inicio
    
    if resultado == 0:
        print(f"\n✅ Concluído com sucesso em {duracao:.2f} segundos.")
        return True
    else:
        print(f"\n❌ ERRO na execução do {nome_script}. Código de saída: {resultado}")
        return False

def main():
    print_header("INICIANDO PIPELINE COMPLETO (PASSOS 1 A 9)")
    tempo_total_inicio = time.time()
    
    # Lista de passos a executar
    passos = [
        ("passo1_master_pipeline.py", "PASSO 1: Extração de Dados Reais (WDI e WGI via API)"),
        ("passo2_master_pipeline.py", "PASSO 2: Análise Exploratória dos Dados Brutos"),
        ("passo2_1_master_pipeline.py", "PASSO 2.1: Limpeza, Agregação (3 Métodos) e EDA Agregados"),
        ("passo3_feat_eng_pipeline.py", "PASSO 3: Engenharia de Features (A1, A2, A3)"),
        ("passo4_model_train_pipeline.py", "PASSO 4: Treinamento de Modelos"),
        ("passo5_eval_pipeline.py", "PASSO 5: Avaliação de Performance"),
        ("passo6_strategy_pipeline.py", "PASSO 6: Análise de Estratégias"),
        ("passo7_shap_pipeline.py", "PASSO 7: Interpretabilidade (SHAP)"),
        ("passo8_geo_pipeline.py", "PASSO 8: Análise Geográfica (SARIMAX)"),
        ("passo9_advanced_pipeline.py", "PASSO 9: Análises Avançadas")
    ]
    
    # Executar cada passo sequencialmente
    for script, descricao in passos:
        if not os.path.exists(script):
            print(f"⚠️ AVISO: Script {script} não encontrado. Saltando...")
            continue
            
        sucesso = executar_passo(script, descricao)
        if not sucesso:
            print("\n🛑 Pipeline interrompido devido a erro.")
            sys.exit(1)
            
    tempo_total_fim = time.time()
    duracao_total = (tempo_total_fim - tempo_total_inicio) / 60
    
    print_header(f"PIPELINE COMPLETO FINALIZADO COM SUCESSO EM {duracao_total:.2f} MINUTOS! 🎉")

if __name__ == "__main__":
    main()
