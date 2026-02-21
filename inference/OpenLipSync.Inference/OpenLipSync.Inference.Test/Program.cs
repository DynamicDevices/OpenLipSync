namespace OpenLipSync.Inference.Test;

using System.Text;
using OpenLipSync.Inference;
using OpenLipSync.Inference.Audio;
using OpenLipSync.Inference.OVRCompat;

class Program
{
    static void Main(string[] args)
    {
        // Optional: first argument = path to WAV file (e.g. from training data). Otherwise auto-pick from dev-clean.
        var wavArg = args.Length > 0 ? args[0].Trim() : null;
        var sample = SampleLocator.FindSample(wavArg);

        if (sample is null)
        {
            Console.WriteLine("No sample found. Either pass a WAV path or ensure training/data/prepared/dev-clean exists.");
            return;
        }

        Console.WriteLine("Selected sample:");
        Console.WriteLine($"  WAV:   {sample.WavPath}");
        if (sample.LabPath != null) Console.WriteLine($"  LAB:   {sample.LabPath}");
        if (sample.JsonPath != null) Console.WriteLine($"  JSON:  {sample.JsonPath}");

        if (!File.Exists(sample.WavPath))
        {
            Console.WriteLine("WAV file missing.");
            return;
        }

        try
        {
            var (audioMono, sampleRate) = ReadWavMono(sample.WavPath);

            // Model expects 16 kHz; training uses per-utterance normalization (mean, std) over mel features.
            const int modelSampleRate = 16000;
            float[] audio16k = sampleRate == modelSampleRate ? audioMono : ResampleAudio(audioMono, sampleRate, modelSampleRate);

            float? normMean = null;
            float? normStd = null;
            using (var backendForStats = new OpenLipSyncBackend())
            {
                var bufferSize = ComputeBufferSizeFrom48kWindow(modelSampleRate, 1024);
                using var emulatorForStats = new ResoniteVisemeAnalyzerEmulator(backendForStats, modelSampleRate, bufferSize);
                if (emulatorForStats.IsInitialized)
                {
                    (normMean, normStd) = backendForStats.ComputeNormalizationStats(audio16k, modelSampleRate);
                    Console.WriteLine($"Per-utterance mel normalization: mean={normMean.Value:0.00} std={normStd.Value:0.00}");
                }
            }

            // Use a fixed time window: 1024 samples at 48kHz (~21.33ms) ⇒ fewer samples at lower rates
            const int windowSamplesAt48k = 1024;

            Console.WriteLine("\nRunning unified scenarios with comparable window (~21.33ms)...");

            double nativeKhz = sampleRate / 1000.0;
            string nativeLabel = $"{nativeKhz:0.#}kHz";
            var exportData = new List<(double timeMs, float[] visemes)>();
            int nativeFrames = RunScenario(
                audioMono,
                sampleRate,
                targetSampleRate: sampleRate,
                windowSamplesAt48k: windowSamplesAt48k,
                label: nativeLabel,
                normMean: normMean,
                normStd: normStd,
                collectForExport: exportData
            );
            Console.WriteLine($"{nativeLabel} completed: {nativeFrames} frames processed");

            int resoniteFrames = RunScenario(
                audioMono,
                sampleRate,
                targetSampleRate: 48000,
                windowSamplesAt48k: windowSamplesAt48k,
                label: "48kHz",
                normMean: normMean,
                normStd: normStd
            );
            Console.WriteLine($"48kHz completed: {resoniteFrames} frames processed");

            if (exportData.Count > 0)
            {
                var outDir = Path.GetDirectoryName(sample.WavPath) ?? ".";
                var baseName = Path.GetFileNameWithoutExtension(sample.WavPath);
                var csvPath = Path.Combine(outDir, $"visemes_{baseName}.csv");
                var htmlPath = Path.Combine(outDir, $"visemes_{baseName}.html");
                WriteVisemesCsv(csvPath, exportData, VisemeNames);
                WriteVisemesHtml(htmlPath, exportData, VisemeNames, baseName);
                Console.WriteLine($"\nVisualization: {htmlPath}");
                Console.WriteLine($"Data export:  {csvPath}");
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Error: {ex.Message}");
        }
    }


    private static float[] ResampleAudio(float[] input, int inputSampleRate, int outputSampleRate)
    {
        if (inputSampleRate == outputSampleRate)
            return input;

        using var resampler = new AudioResampler(inputSampleRate, outputSampleRate);
        return resampler.Resample(input);
    }

    private static int ComputeBufferSizeFrom48kWindow(int sampleRate, int windowSamplesAt48k)
    {
        return (int)Math.Round(sampleRate * (double)windowSamplesAt48k / 48000.0);
    }

    private static readonly string[] VisemeNames = { "sil", "PP", "FF", "TH", "DD", "kk", "CH", "SS", "nn", "RR", "aa", "E", "ih", "oh", "ou" };

    private static int RunScenario(float[] originalAudio, int originalSampleRate, int targetSampleRate, int windowSamplesAt48k, string label, float? normMean = null, float? normStd = null, List<(double timeMs, float[] visemes)>? collectForExport = null)
    {
        int bufferSize = ComputeBufferSizeFrom48kWindow(targetSampleRate, windowSamplesAt48k);
        using var backend = new OpenLipSyncBackend();
        using var emulator = new ResoniteVisemeAnalyzerEmulator(backend, targetSampleRate, bufferSize);

        if (!emulator.IsInitialized)
        {
            Console.WriteLine($"{label}: initialization failed (check if ONNX model exists in export/ directory)");
            return 0;
        }

        if (normMean.HasValue && normStd.HasValue)
            backend.SetPerUtteranceNormalization(normMean.Value, normStd.Value);

        var audio = targetSampleRate == originalSampleRate
            ? originalAudio
            : ResampleAudio(originalAudio, originalSampleRate, targetSampleRate);

        double frameDurationMs = 1000.0 * bufferSize / targetSampleRate;

        Console.WriteLine($"{label}: buffer={bufferSize} samples (~{frameDurationMs:0.00} ms)");

        int totalFrames = (audio.Length + bufferSize - 1) / bufferSize;
        var sampleAtFrames = new[] { 0, totalFrames / 4, totalFrames / 2, (3 * totalFrames) / 4, Math.Max(0, totalFrames - 1) }.Distinct().OrderBy(f => f).ToArray();

        int n = ProcessFrames(emulator, audio, bufferSize, sampleAtFrames, frameDurationMs, label, VisemeNames, collectForExport);

        return n;
    }

    private static int ProcessFrames(
        ResoniteVisemeAnalyzerEmulator emulator,
        float[] audio,
        int bufferSize,
        int[] sampleAtFrames,
        double frameDurationMs,
        string label,
        string[] visemeNames,
        List<(double timeMs, float[] visemes)>? collectForExport = null)
    {
        int processedFrames = 0;
        var set = new HashSet<int>(sampleAtFrames);

        for (int offset = 0; offset < audio.Length; offset += bufferSize)
        {
            int remaining = audio.Length - offset;
            int count = Math.Min(bufferSize, remaining);

            emulator.OnAudioUpdate(audio.AsSpan(offset, count), bufferSize);

            double tMs = processedFrames * frameDurationMs;
            if (collectForExport != null)
            {
                var vv = new float[15];
                for (int v = 0; v < 15; v++) vv[v] = emulator[v];
                collectForExport.Add((tMs, vv));
            }

            if (set.Contains(processedFrames))
            {
                var parts = new List<string>();
                for (int v = 0; v < visemeNames.Length && v < 15; v++)
                {
                    float val = emulator[v];
                    if (val > 0.01f) parts.Add($"{visemeNames[v]}={val:0.00}");
                }
                Console.WriteLine($"[{label} t={tMs:0.0}ms frame={processedFrames}] {string.Join(" ", parts.Count > 0 ? parts : new[] { "sil=1.00" })}");
            }

            processedFrames++;
        }

        return processedFrames;
    }

    private static void WriteVisemesCsv(string path, List<(double timeMs, float[] visemes)> data, string[] visemeNames)
    {
        const int numVisemes = 15;
        var inv = System.Globalization.CultureInfo.InvariantCulture;
        var sb = new StringBuilder();
        sb.Append("time_ms").Append(',').Append(string.Join(",", visemeNames)).AppendLine();
        foreach (var (timeMs, visemes) in data)
        {
            sb.Append(timeMs.ToString("F2", inv));
            for (int v = 0; v < numVisemes; v++)
            {
                float val = v < visemes.Length ? visemes[v] : 0f;
                sb.Append(',').Append(val.ToString("F4", inv));
            }
            sb.AppendLine();
        }
        // UTF-8 with BOM so Excel detects encoding when opening
        var utf8Bom = new System.Text.UTF8Encoding(encoderShouldEmitUTF8Identifier: true);
        File.WriteAllText(path, sb.ToString(), utf8Bom);
    }

    private static void WriteVisemesHtml(string path, List<(double timeMs, float[] visemes)> data, string[] visemeNames, string title)
    {
        // Build JSON arrays: times (seconds), and one array per viseme
        var times = data.Select(d => d.timeMs / 1000.0).ToArray();
        var series = new List<object>();
        var colors = new[] { "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf", "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5" };
        for (int v = 0; v < visemeNames.Length && v < 15; v++)
        {
            var vals = data.Select(d => (double)d.visemes[v]).ToArray();
            series.Add(new { name = visemeNames[v], color = colors[v % colors.Length], values = vals });
        }
        var json = System.Text.Json.JsonSerializer.Serialize(new { times, series });
        // Embed in application/json script so no escaping can break the chart script
        var html = $@"<!DOCTYPE html>
<html>
<head>
  <meta charset=""utf-8"">
  <title>Visemes – {EscapeHtml(title)}</title>
  <script src=""https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js""></script>
</head>
<body>
  <h1>Visemes – {EscapeHtml(title)}</h1>
  <p>Time (s) vs viseme activation (0–1). Toggle series by clicking the legend.</p>
  <p>
    <button type=""button"" id=""all-on"">All on</button>
    <button type=""button"" id=""all-off"">All off</button>
  </p>
  <div style=""max-width:1200px;height:500px"">
    <canvas id=""chart""></canvas>
  </div>
  <script type=""application/json"" id=""viseme-data"">{json}</script>
  <script>
    const data = JSON.parse(document.getElementById('viseme-data').textContent);
    const datasets = data.series.map((s) => ({{
      label: s.name,
      data: data.times.map((t, j) => ({{ x: t, y: s.values[j] }})),
      borderColor: s.color,
      backgroundColor: s.color + '33',
      borderWidth: 1.5,
      fill: false,
      pointRadius: 0,
      tension: 0.1
    }}));
    const maxTime = data.times.length > 0 ? Math.max(...data.times) : 1;
    const chart = new Chart(document.getElementById('chart'), {{
      type: 'line',
      data: {{ datasets }},
      options: {{
        responsive: true,
        maintainAspectRatio: false,
        scales: {{
          x: {{ type: 'linear', title: {{ display: true, text: 'Time (s)' }}, min: 0, max: maxTime }},
          y: {{ type: 'linear', title: {{ display: true, text: 'Activation' }}, min: 0, max: 1 }}
        }},
        plugins: {{ legend: {{ position: 'top' }} }}
      }}
    }});
    document.getElementById('all-on').addEventListener('click', () => {{
      chart.data.datasets.forEach(d => d.hidden = false);
      chart.update();
    }});
    document.getElementById('all-off').addEventListener('click', () => {{
      chart.data.datasets.forEach(d => d.hidden = true);
      chart.update();
    }});
  </script>
</body>
</html>";
        File.WriteAllText(path, html);
    }

    private static string EscapeHtml(string s)
    {
        return s.Replace("&", "&amp;").Replace("<", "&lt;").Replace(">", "&gt;").Replace("\"", "&quot;");
    }

    // Minimal WAV reader for PCM16 and IEEE float32; returns mono float samples in [-1,1].
    private static (float[] samples, int sampleRate) ReadWavMono(string path)
    {
        using var fs = File.OpenRead(path);
        using var br = new BinaryReader(fs);

        if (new string(br.ReadChars(4)) != "RIFF") throw new InvalidDataException("Not RIFF");
        br.ReadInt32();
        if (new string(br.ReadChars(4)) != "WAVE") throw new InvalidDataException("Not WAVE");

        ushort audioFormat = 0;
        ushort numChannels = 0;
        int sampleRate = 0;
        ushort bitsPerSample = 0;
        int dataSize = 0;
        long dataPos = 0;

        while (fs.Position < fs.Length)
        {
            string id = new string(br.ReadChars(4));
            int size = br.ReadInt32();
            long next = fs.Position + size;

            if (id == "fmt ")
            {
                audioFormat = br.ReadUInt16();
                numChannels = br.ReadUInt16();
                sampleRate = br.ReadInt32();
                br.ReadInt32();
                br.ReadUInt16();
                bitsPerSample = br.ReadUInt16();
                fs.Position = next;
            }
            else if (id == "data")
            {
                dataPos = fs.Position;
                dataSize = size;
                break;
            }
            else
            {
                fs.Position = next;
            }
        }

        if (dataPos == 0) throw new InvalidDataException("Missing data chunk");
        if (numChannels == 0 || sampleRate == 0) throw new InvalidDataException("Invalid fmt chunk");

        fs.Position = dataPos;

        int bytesPerSample = bitsPerSample / 8;
        int totalSamplesPerChannel = dataSize / (bytesPerSample * numChannels);

        float[] mono = new float[totalSamplesPerChannel];

        switch (audioFormat)
        {
            case 1:
                if (bitsPerSample != 16) throw new NotSupportedException($"PCM {bitsPerSample}b not supported");
                for (int i = 0; i < totalSamplesPerChannel; i++)
                {
                    int left = br.ReadInt16();
                    int right = numChannels == 2 ? br.ReadInt16() : left;
                    mono[i] = ((left + right) * 0.5f) / 32768f;
                }
                break;

            case 3:
                if (bitsPerSample != 32) throw new NotSupportedException($"Float {bitsPerSample}b not supported");
                for (int i = 0; i < totalSamplesPerChannel; i++)
                {
                    float left = br.ReadSingle();
                    float right = numChannels == 2 ? br.ReadSingle() : left;
                    mono[i] = (left + right) * 0.5f;
                }
                break;

            default:
                throw new NotSupportedException($"Audio format {audioFormat} not supported");
        }

        return (mono, sampleRate);
    }


}