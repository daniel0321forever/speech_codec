# Readme

簡單寫點readme。

## Dataset架構

Dataset的大資料夾底下，有一堆小資料夾，小資料夾的名稱代表語者的代號，每個小資料夾裡面都包含同一個語者的一大堆音檔。

## 抽取特徵

```
 python dataset.py [dataset_dir] [output_path]
```

dataset_dir: dataset大資料夾的位置

output_path: 存特徵檔的地方

這一步抽取log-Mel spectrogram。我的實驗做在log-Mel的feature space上面。這點很重要。因為當我在這個feature space做encode/decode之後，最後還需要一步，把log-Mel spectrogram還原回去waveform。這邊需要靠neural vocoder幫忙。

可以去這邊抓pretrained vocoder：

https://github.com/kan-bayashi/ParallelWaveGAN

要注意不同pretrained vocoder的input/output形式不同。雖然同樣都是log-Mel spec -> waveform，但是sample rate/Mel range/......有一大堆參數不同，要注意。

我這邊code的格式是sample_rate (Fs) = 24000; Mel range = 80-7600; fft_size (FFT) = 2048; hop_length (Hop) = 300; win_length (Win) = 1200。同樣spec的所有pretrained vocoder都能拿來用。我當初是直接抓[vctk_parallel_wavegan.v1](https://drive.google.com/open?id=1dGTu-B7an2P5sEOepLPjpOaasgaSnLpi) 下來用，然後用了我之前已經寫過的一個class PWGVocoder()作為一個小小的wrapper去inference。HifiGAN效果更好，但是要跑更久。在資料壓縮上面，處理時間也是個重要的標準。

實務上來說這個vocoder只有在最後測試的時候才會用到。Training/validation都是測在log-Mel feature space上面。但最後評估壓縮好壞的時候，還是要回到waveform fomain來測，這時候就需要vocoder。

## Training

```
python train.py
```

gpu_id、model_dir、logger_path要在code裡面自己調整。第一個是使用的gpu的id，第二個是存model checkpoint的資料夾，logger_path是存tensorboard logger的路徑。

另外train_dataset跟test_dataset的路徑也要自己根據特徵的路徑而調整。預設是train_spc.pkl跟test_spc.pkl。

## Inference

```
python test.py [model_path] [source_audio] [output_prefix] [gpu_id]
```

model_path: 要測試的模型的路徑

source_audio: 要壓縮的音檔的路徑

output_prefix: 這份code會一次生出4個不同版本的encode再decode之後的結果，分別是用了一個VQ、兩個VQ、三個VQ encode的結果，還有直接抽log-Mel spectrogram候用vocoder重建的結果。這個參數決定這四個檔案的prefix要是什麼。

gpu_id: 使用的gpu的id，如果不想用gpu的話可以給-1

另外這份code還會去算PESQ這個客觀的標準。這個標準我自己覺得很嚴格，但是那確實是一個比較常用的標準。
