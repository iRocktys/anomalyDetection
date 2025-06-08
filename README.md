# AnomalyDetection

Repositório de apoio ao projeto de monografia de Leandro Martins Tosta, desenvolvido como parte do trabalho de conclusão de curso em Ciência da Computação. O projeto propõe a aplicação de técnicas de aprendizado profundo (Deep Learning) e aprendizado de máquina (Machine Learning) para detecção de tráfego anômalo em redes de computadores, com foco especial em ataques de negação de serviço distribuído (DDoS) presentes no dataset CICDDoS2019.

## Objetivo

Implementar e avaliar modelos de detecção de intrusão utilizando diferentes arquiteturas de redes neurais — CNN, LSTM e uma arquitetura híbrida CNN-LSTM com classificação final via SVM — a fim de identificar padrões anômalos em tráfego de rede. As abordagens são comparadas por meio de métricas de avaliação como *accuracy*, *precision*, *recall*, *F1-score*, *confusion matrix* e *ROC AUC*.

## Arquiteturas Implementadas

- CNN (Convolutional Neural Network): Extração de padrões espaciais com múltiplas camadas convolucionais e operações de max pooling.
- LSTM (Long Short-Term Memory): Captura de dependências temporais em sequências de tráfego de rede.
- Modelo Híbrido CNN+LSTM+SVM:
  - Extração de características com CNN.
  - Modelagem temporal com múltiplas camadas LSTM.
  - Redução de dimensionalidade via PCA.
  - Classificação final com SVM (Support Vector Machine).

## Estrutura do Repositório

CNN.py                # Arquitetura da rede convolucional  
LSTM.py               # Arquitetura da rede LSTM  
Hybrid.py             # Arquitetura híbrida CNN+LSTM com SVM  
Sequence.py           # Classe de carregamento e preparação dos dados  
CompileModels.ipynb   # Pipeline de execução e treinamento  
output/               # Diretório para salvar modelos e artefatos (não incluído)  
README.md             # Este documento  

## Execução

Todos os experimentos foram conduzidos com suporte à GPU, utilizando CUDA. O treinamento é realizado por meio do Jupyter Notebook, com monitoramento de métricas de validação e aplicação de early stopping para evitar sobreajuste.

Exemplo de execução:

jupyter notebook CompileModels.ipynb

## Tecnologias Utilizadas

- Python 3.9  
- PyTorch  
- Scikit-learn  
- NumPy  
- Pandas  
- Matplotlib  
- Seaborn  
- CUDA Toolkit  

## Licença

Este repositório é parte integrante de uma monografia acadêmica e está disponível exclusivamente para fins educacionais e de pesquisa. O autor se reserva o direito de atualizar ou modificar os conteúdos conforme necessário.
