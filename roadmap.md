# Roteiro: TTS (Text-to-Speech) Leve do Zero em PT-BR

Este guia estrutura o aprendizado e a implementação de um sintetizador de voz neural do zero, focado em **baixo custo computacional** (inferência rápida em CPUs fracas) e **treinamento viável em hardware modesto**.

---

## 1. Visão Geral da Arquitetura Escolhida

Para rodar leve e sem engasgos, evitamos redes autoregressivas lentas (que geram um pedaço de áudio por vez dependendo do anterior). Usamos um pipeline em **2 estágios não-autoregressivos**:

```text
[Texto PT-BR] ──► (G2P / Fonemas) ──► [Modelo Acústico] ──► [Mel-Espectrograma] ──► [Vocoder] ──► [Áudio .wav]
                                      (Estilo FastSpeech2)         (Matriz 2D)       (HiFi-GAN V3)
```

1. **Modelo Acústico (Texto $\rightarrow$ Espectrograma):** Converte fonemas em uma matriz 2D de frequências ao longo do tempo. Usa um *Duration Predictor* para saber quanto tempo dura cada som, sem precisar de RNNs.
2. **Vocoder (Espectrograma $\rightarrow$ Onda Sonora):** Converte a matriz 2D de volta em arquivo de áudio (`.wav`) usando apenas Convoluções 1D Transpostas (`ConvTranspose1D`).

---

## 2. O Caminho das Pedras (Etapas de Estudo e Código)

### Fase 1: Processamento de Sinal (DSP) e PyTorch
Antes de criar qualquer camada neural, domine a conversão entre som e tensor.
* [ ] **Conceitos:** *Sample Rate* (use **22.050 Hz** como padrão), *STFT* (Short-Time Fourier Transform), *Hop Length* (ex: 256), *Win Length* (ex: 1024) e *Escala Mel* (80 canais/bandas).
* [ ] **Prática:** Escrever um script usando `torchaudio.transforms.MelSpectrogram` que lê um `.wav` e salva um tensor de shape `(80, T)`.
* [ ] **Teste de Sanidade:** Usar o algoritmo clássico **Griffin-Lim** (`torchaudio.transforms.GriffinLim`) para transformar esse espectrograma de volta em `.wav`. Vai soar metálico/robótico, mas prova que seus parâmetros de áudio estão corretos.

### Fase 2: Construindo e Treinando o Vocoder (HiFi-GAN Leve)
Comece pelo fim! O Vocoder não precisa de texto nem alinhamento, apenas pares de `(Mel-Espectrograma Real, Áudio Real)`.
* [ ] **Arquitetura:** Implementar o gerador do **HiFi-GAN** na configuração **V3** (versão ultra-leve com menos canais ocultos) e os discriminadores (MPD - *Multi-Period Discriminator* e MSD - *Multi-Scale Discriminator*).
* [ ] **Otimização de Leveza:** Substituir convoluções 1D padrão por *Depthwise Separable Convolutions* nos blocos residuais (MRF) para cortar o tamanho do modelo pela metade.
* [ ] **Critério de Sucesso:** Dar um Mel-Espectrograma extraído de um áudio real para a sua rede e ouvir uma voz humana limpa e natural sair do outro lado.

### Fase 3: Preparação de Texto (G2P - Grapheme-to-Phoneme)
Em português, letras mudam de som dependendo do contexto (ex: o "x" em *táxi*, *exame* e *xícara*; o "e" no final das palavras virando som de "i").
* [ ] **Por que usar fonemas:** Treinar com fonemas em vez de letras puras faz a rede convergir 3x mais rápido e exige um modelo muito menor, pois ela não precisa "decorar" regras de ortografia.
* [ ] **Prática:** Criar um limpador de texto (converter números como `10` para `dez`, remover pontuação estranha) e usar uma ferramenta G2P para PT-BR (ex: `phonemizer` com backend `espeak-ng` configurado para `pt-br`, ou a biblioteca `g2p_en`/`gruut`) para transformar texto em símbolos fonéticos (IPA).

### Fase 4: Alinhamento Forçado (Durações)
Para treinar um modelo não-autoregressivo, você precisa saber exatamente quantos *frames* de áudio cada fonema dura nos seus arquivos de treino.
* [ ] **Prática:** Rodar o **Montreal Forced Aligner (MFA)** usando o modelo acústico e dicionário de português já disponíveis na ferramenta sobre o seu dataset.
* [ ] **Saída:** Arquivos `.TextGrid` que dizem: *"o fonema /a/ começou em 0.12s e terminou em 0.20s"*. Converta esses tempos para número de frames (`tempo * sample_rate / hop_length`).

### Fase 5: Modelo Acústico Leve (Estilo FastSpeech 2)
Agora você conecta o texto ao espectrograma gerado na Fase 1.
* [ ] **Camadas:**
  1. *Encoder:* Embedding dos fonemas + 3 ou 4 blocos leves de Conv1D + Self-Attention (dimensão oculta pequena, ex: `128` ou `192`).
  2. *Variance Adaptor:* Preditor de Duração (2 camadas Conv1D) + Preditor de Pitch ($F_0$ extraído com `pyworld` ou `librosa.yin`).
  3. *Length Regulator:* Repete os vetores do encoder conforme a duração prevista.
  4. *Decoder:* Mais 3 ou 4 blocos de Conv1D + camada Linear final projetando para `80` canais (o Mel-Espectrograma).
* [ ] **Loss Function:** `L1Loss` (MAE) entre o espectrograma previsto e o real + `MSELoss` para a duração (em escala logarítmica) e pitch.

### Fase 6: Empacotamento para Computadores Fracos
* [ ] **Juntar as pontas:** Texto $\rightarrow$ Fonemas $\rightarrow$ Modelo Acústico $\rightarrow$ Vocoder $\rightarrow$ `.wav`.
* [ ] **Exportação:** Converter ambos os modelos para **ONNX** (`torch.onnx.export`) e aplicar quantização dinâmica (`INT8`) via `onnxruntime`. O modelo final terá menos de **25 MB** e rodará em tempo real até em processadores antigos.

---

## 3. Melhores Datasets Abertos em Português Brasileiro (PT-BR)

Para TTS, **qualidade de gravação e voz única (Single-Speaker)** valem muito mais do que milhares de horas com ruído de fundo.

### Para o Modelo Acústico (Precisa de voz limpa, sem eco e transcrição fiel)
1. **TTS-Portuguese Corpus (Edresson Casanova)**
   * **O que é:** ~10 horas de áudio de um único locutor masculino, gravado em ambiente controlado, já formatado estilo LJSpeech.
   * **Por que usar:** É o melhor dataset aberto para treinar um modelo acústico PT-BR do zero sem precisar limpar ruídos ou separar locutores.
2. **BRSpeech / CETUC (Subconjunto Single Speaker)**
   * **O que é:** Diversos locutores brasileiros lendo frases foneticamente balanceadas.
   * **Como usar:** Filtre e pegue apenas **1 locutor** com a gravação mais limpa para o treino inicial do modelo acústico.
3. **Common Voice (Mozilla - Português)**
   * **Atenção:** Tem muito ruído de microfone caseiro e vários locutores diferentes. **Não use** para treinar a voz final do modelo acústico no início, mas pode ser útil depois se quiser testar multi-speaker.

### Para o Vocoder (Pode misturar várias vozes!)
O Vocoder só aprende a reconstruir ondas sonoras a partir de frequências; ele não precisa saber quem está falando nem o que está sendo dito.
* **Estratégia:** Junte o **TTS-Portuguese Corpus** + áudios limpos do **MLS (Multilingual LibriSpeech - Português)** + **BRSpeech**. Ter várias vozes masculinas e femininas treinando o Vocoder deixa ele robusto para sintetizar qualquer timbre sem distorções.

---

## 4. Especificações Sugeridas para "Leveza Extrema"

Se o foco é rodar em máquinas muito fracas, trave estas configurações desde o dia 1:

| Parâmetro | Valor Recomendado | Motivo |
| :--- | :--- | :--- |
| **Sample Rate** | `22050 Hz` | Metade do custo computacional de `44100 Hz`, mantendo clareza total para fala humana. |
| **Canais Mel (`n_mels`)** | `80` | Padrão da indústria; reduzido o suficiente para matrizes pequenas. |
| **Dimensão Oculta (Acústico)** | `128` ou `192` | Modelos grandes usam `256` ou `512`. `192` reduz os parâmetros em mais de 50%. |
| **Canais Iniciais (Vocoder)** | `128` ou `256` | Configuração *HiFi-GAN V3* (a versão V1 usa `512` e é pesada para CPU). |
