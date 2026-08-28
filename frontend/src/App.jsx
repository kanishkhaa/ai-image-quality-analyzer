
import { useState } from "react";

const API_URL = import.meta.env.VITE_API_URL;
function App() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handleFileChange = (event) => {
    const selectedFile = event.target.files[0];

    if (!selectedFile) return;

    if (!selectedFile.type.startsWith("image/")) {
      setError("Please select a valid image file.");
      return;
    }

    setFile(selectedFile);
    setPreview(URL.createObjectURL(selectedFile));
    setResult(null);
    setError("");
  };

  const analyzeImage = async () => {
    if (!file) {
      setError("Please select an image first.");
      return;
    }

    setLoading(true);
    setError("");
    setResult(null);

    try {
      const formData = new FormData();
      formData.append("file", file);

      const response = await fetch(`${API_URL}/api/analyze`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => null);

        throw new Error(
          errorData?.detail || `Server error: ${response.status}`
        );
      }

      const data = await response.json();

      console.log("Analysis result:", data);
      setResult(data);
    } catch (err) {
      console.error(err);

      setError(err.message || "Failed to analyze image.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-slate-950 text-white">
      {/* Header */}
      <header className="border-b border-slate-800 bg-slate-950/95">
        <div className="mx-auto flex max-w-7xl items-center justify-between px-6 py-5">
          <div>
            <h1 className="text-2xl font-bold tracking-tight">
              Image Quality AI
            </h1>

            <p className="mt-1 text-sm text-slate-400">
              AI-powered image quality & defect detection
            </p>
          </div>

          <div className="flex items-center gap-2 rounded-full border border-emerald-500/20 bg-emerald-500/10 px-4 py-2 text-sm text-emerald-400">
            <span className="h-2 w-2 rounded-full bg-emerald-400"></span>
            AI Engine Online
          </div>
        </div>
      </header>

      {/* Main */}
      <main className="mx-auto max-w-7xl px-6 py-10">
        <div className="grid gap-8 lg:grid-cols-2">
          {/* Upload Card */}
          <section className="rounded-2xl border border-slate-800 bg-slate-900 p-6 shadow-xl">
            <h2 className="text-xl font-semibold">
              Analyze an Image
            </h2>

            <p className="mt-2 text-sm text-slate-400">
              Upload an image to evaluate its visual quality.
            </p>

            {/* Upload */}
            <label className="mt-6 flex cursor-pointer flex-col items-center justify-center rounded-xl border-2 border-dashed border-slate-700 bg-slate-950 px-6 py-12 text-center transition hover:border-blue-500 hover:bg-slate-900">
              <input
                type="file"
                accept="image/*"
                onChange={handleFileChange}
                className="hidden"
              />

              <div className="mb-4 flex h-14 w-14 items-center justify-center rounded-full bg-blue-500/10 text-3xl text-blue-400">
                ↑
              </div>

              <strong className="text-base">
                {file ? file.name : "Choose an image"}
              </strong>

              <span className="mt-2 text-sm text-slate-500">
                JPG, PNG, WEBP or BMP
              </span>
            </label>

            {/* Preview */}
            {preview && (
              <div className="mt-6 overflow-hidden rounded-xl border border-slate-800 bg-slate-950">
                <img
                  src={preview}
                  alt="Selected"
                  className="max-h-80 w-full object-contain"
                />
              </div>
            )}

            {/* Error */}
            {error && (
              <div className="mt-5 rounded-lg border border-red-500/20 bg-red-500/10 px-4 py-3 text-sm text-red-400">
                {error}
              </div>
            )}

            {/* Button */}
            <button
              onClick={analyzeImage}
              disabled={!file || loading}
              className="mt-6 w-full rounded-xl bg-blue-600 px-5 py-3.5 font-semibold text-white transition hover:bg-blue-500 disabled:cursor-not-allowed disabled:opacity-40"
            >
              {loading ? "Analyzing..." : "Analyze Image"}
            </button>
          </section>

          {/* Result Card */}
          {result ? (
            <section className="rounded-2xl border border-slate-800 bg-slate-900 p-6 shadow-xl">
              {/* Result Header */}
              <div className="flex items-start justify-between gap-6">
                <div>
                  <h2 className="text-xl font-semibold">
                    Analysis Result
                  </h2>

                  <p className="mt-2 text-sm text-slate-400">
                    Image quality assessment
                  </p>
                </div>

                <div className="text-right">
                  <div className="text-4xl font-bold text-blue-400">
                    {Number(result.quality_score ?? 0).toFixed(1)}
                  </div>

                  <div className="text-sm text-slate-500">
                    /100
                  </div>
                </div>
              </div>

              {/* Quality */}
              <div className="mt-6 rounded-xl border border-slate-800 bg-slate-950 p-4">
                <div className="flex items-center justify-between">
                  <span className="text-sm text-slate-400">
                    Overall Quality
                  </span>

                  <strong className="text-emerald-400">
                    {result.quality_label}
                  </strong>
                </div>
              </div>

              {/* Issues */}
              <h3 className="mt-8 text-base font-semibold">
                Detected Issues
              </h3>

              {result.issues && result.issues.length > 0 ? (
                <div className="mt-4 space-y-3">
                  {result.issues.map((issue, index) => (
                    <div
                      className="flex items-center justify-between rounded-xl border border-slate-800 bg-slate-950 p-4"
                      key={index}
                    >
                      <div>
                        <strong className="block text-sm">
                          {issue.type
                            .replaceAll("_", " ")
                            .toUpperCase()}
                        </strong>

                        <span className="mt-1 block text-xs text-slate-500">
                          Confidence:{" "}
                          {(issue.confidence * 100).toFixed(0)}%
                        </span>
                      </div>

                      <span
                        className={`rounded-full px-3 py-1 text-xs font-semibold ${
                          issue.severity === "high"
                            ? "bg-red-500/10 text-red-400"
                            : issue.severity === "medium"
                            ? "bg-yellow-500/10 text-yellow-400"
                            : "bg-green-500/10 text-green-400"
                        }`}
                      >
                        {issue.severity}
                      </span>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="mt-4 rounded-xl border border-emerald-500/20 bg-emerald-500/10 p-4 text-sm text-emerald-400">
                  No significant quality issues detected.
                </div>
              )}

              {/* Statistics */}
              {result.statistics && (
                <>
                  <h3 className="mt-8 text-base font-semibold">
                    Image Statistics
                  </h3>

                  <div className="mt-4 grid grid-cols-2 gap-3">
                    {Object.entries(result.statistics).map(
                      ([key, value]) => (
                        <div
                          className="rounded-xl border border-slate-800 bg-slate-950 p-4"
                          key={key}
                        >
                          <span className="block text-xs text-slate-500">
                            {key.replaceAll("_", " ").toUpperCase()}
                          </span>

                          <strong className="mt-1 block text-lg">
                            {typeof value === "number"
                              ? value.toFixed(2)
                              : value}
                          </strong>
                        </div>
                      )
                    )}
                  </div>
                </>
              )}

              {/* Explanation */}
              {result.explanation && (
                <>
                  <h3 className="mt-8 text-base font-semibold">
                    Explanation
                  </h3>

                  <div className="mt-4 space-y-2 rounded-xl border border-slate-800 bg-slate-950 p-4">
                    {result.explanation.map((text, index) => (
                      <p
                        key={index}
                        className="text-sm leading-6 text-slate-400"
                      >
                        • {text}
                      </p>
                    ))}
                  </div>
                </>
              )}
            </section>
          ) : (
            <section className="flex min-h-[500px] items-center justify-center rounded-2xl border border-slate-800 bg-slate-900 p-6 text-center shadow-xl">
              <div>
                <div className="mx-auto mb-5 flex h-16 w-16 items-center justify-center rounded-full bg-slate-800 text-2xl">
                  ✦
                </div>

                <h2 className="text-xl font-semibold">
                  Ready for Analysis
                </h2>

                <p className="mt-2 max-w-sm text-sm leading-6 text-slate-500">
                  Upload an image and click Analyze Image to see
                  AI-powered quality assessment results.
                </p>
              </div>
            </section>
          )}
        </div>
      </main>
    </div>
  );
}

export default App;
