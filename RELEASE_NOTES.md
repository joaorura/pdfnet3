# Relatório de Lançamento — pDFNet3 v1.2.0

## 🎯 Objetivo da Versão
Esta versão atualiza os pesos dos decoders do **pDFNet3 (DeepFilterNet3 com FiLM)** para supressão robusta de ruídos acústicos de notebook (ventoinhas) e forte atenuação de vazamentos de vozes de terceiros ao fundo captados por microfones de computadores e headsets.

---

## 📊 Avaliação e Métricas de Desempenho

Resultados avaliados em conjunto de teste holding-out (fatias acústicas reais segregadas não vistas durante o treino):

| Cenário Acústico / Tipo de Ruído | Supressão de Ruído Puro | Ganho de SNR ($\Delta$SNR) | Ganho de Inteligibilidade ($\Delta$STOI) | Ativação do Filtro Neural (DF/ERB) |
| :--- | :---: | :---: | :---: | :---: |
| **Vozes ao Fundo (Microfone Externo / Headset)** | **31.96 dB** | **+4.04 dB** | **+0.073** | **83.3%** |
| **Vozes ao Fundo (Microfone Embutido / Notebook)** | **29.12 dB** | **+4.46 dB** | **+0.072** | **85.5%** |
| **Ruído Acústico Notebook (Ventoinha / CPU)** | **24.73 dB** | **+6.20 dB** | **+0.160** | **82.0%** |
| **Ruído de Cafeteria e Babble Multifalante** | **27.48 dB** | **+5.82 dB** | **+0.125** | **85.0%** |

---

## 🛠️ Mudanças Arquiteturais e de Treinamento
- **Estratégia de Congelamento:** O encoder convolucional (`enc.onnx`) permaneceu congelado para preservar a invariância acústica universal, enquanto os decoders ERB e DF receberam fine-tuning supervisionado via AdamW.
- **Treinamento Multi-Dataset:** Base expandida contendo 8.495 amostras de ruídos locais, ruídos combinados com RIR (reverberação de sala) e babble multi-falante sintético em português.
- **Early Stopping:** Parada antecipada acionada no passo 1.200 (melhor checkpoint registrado no passo 600), atingindo perda de validação de **1.1810**.
- **Equilíbrio FiLM:** Treinado com dropout estocástico de perfil de voz ($p=0.50$), garantindo excelente performance como filtro acústico padrão quando o perfil de voz não está ativo.
