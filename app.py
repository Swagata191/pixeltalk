import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import plotly.graph_objects as go

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="PixelTalk", page_icon="🔢", layout="wide")

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Space Mono', monospace;
    background-color: #040209;
    color: #cbd5e1;
}
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.5rem; padding-bottom: 3rem; }

.hero { text-align: center; padding: 2rem 0 1.2rem; }
.hero-label {
    font-size: 0.65rem; letter-spacing: 0.35em; color: #a855f7;
    text-transform: uppercase; margin-bottom: 0.6rem;
}
.hero-title {
    font-family: 'Space Mono', monospace;
    font-size: clamp(2rem, 5vw, 3.4rem); font-weight: 700; line-height: 1.05;
    background: linear-gradient(135deg, #ffffff 0%, #a855f7 50%, #06b6d4 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text; margin: 0; letter-spacing: -0.02em;
}
.hero-sub { font-size: 0.72rem; color: #4b4b74; margin-top: 0.7rem; letter-spacing: 0.04em; }

.divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(168, 85, 247, 0.25), transparent);
    margin: 1.2rem 0;
}

[data-testid="stFileUploader"] {
    background: #090611; border: 1.5px dashed #221936;
    border-radius: 8px; padding: 1rem; transition: border-color 0.25s;
}
[data-testid="stFileUploader"]:hover { border-color: #06b6d4; }
[data-testid="stFileUploader"] label {
    font-family: 'Space Mono', monospace !important;
    font-size: 0.7rem !important; color: #4b4b74 !important; letter-spacing: 0.05em;
}

.result-card {
    background: linear-gradient(135deg, #090611 0%, #030207 100%);
    border: 1px solid #221936; border-radius: 12px;
    padding: 1.6rem; margin-top: 1rem; position: relative; overflow: hidden;
}
.result-card::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, #4c1d95, #06b6d4);
}
.result-label { font-size: 0.6rem; letter-spacing: 0.3em; color: #4b4b74; text-transform: uppercase; margin-bottom: 0.3rem; }
.result-digit {
    font-family: 'Space Mono', monospace; font-size: 5rem; font-weight: 700; line-height: 1;
    background: linear-gradient(135deg, #ffffff, #a855f7);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
}
.result-confidence { font-size: 0.68rem; color: #06b6d4; margin-top: 0.5rem; letter-spacing: 0.05em; }

.prob-row { display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.34rem; }
.prob-digit-label { font-family: 'Space Mono', monospace; font-size: 0.7rem; font-weight: 700; color: #94a3b8; width: 14px; text-align: right; }
.prob-bar-bg { flex: 1; height: 6px; background: #130d22; border-radius: 2px; overflow: hidden; }
.prob-bar-fill { height: 100%; border-radius: 2px; background: linear-gradient(90deg, #4c1d95, #06b6d4); }
.prob-pct { font-size: 0.58rem; color: #4b4b74; width: 38px; text-align: right; }

.section-title { font-size: 0.58rem; letter-spacing: 0.28em; color: #3b3b5c; text-transform: uppercase; margin-bottom: 0.7rem; }
.chart-label { font-size: 0.52rem; letter-spacing: 0.2em; color: #3b3b5c; text-transform: uppercase; margin-bottom: 0.3rem; text-align: center; }

[data-testid="stImage"] img { border-radius: 6px; border: 1px solid #221936; image-rendering: pixelated; }

.footer { text-align: center; margin-top: 2rem; font-size: 0.55rem; color: #221d33; letter-spacing: 0.1em; }

.placeholder-box {
    text-align: center; padding: 3rem 1rem;
    color: #3b3b5c; font-size: 0.65rem; letter-spacing: 0.08em; line-height: 2.2;
}
</style>
""", unsafe_allow_html=True)

# ── Plotly helpers ────────────────────────────────────────────────────────────
PAPER  = "#040209"        
PLOT   = "#07040e"
GRID   = "#332a4b"
FONT   = "Space Mono, monospace"
PURPLE = "#a855f7"       
CYAN   = "#06b6d4"
DARK_PURPLE = "#4c1d95"

def base_layout(h=240):
    return dict(
        paper_bgcolor=PAPER, plot_bgcolor=PLOT,
        font=dict(family=FONT, color="#4b4b74", size=8),
        margin=dict(l=10, r=10, t=30, b=10),
        height=h,
    )

def make_donut(probs, pred):
    colors = [PURPLE if i == pred else "#090611" for i in range(10)]
    borders = [CYAN if i == pred else "#221936" for i in range(10)]
    fig = go.Figure(go.Pie(
        labels=[str(i) for i in range(10)],
        values=probs,
        hole=0.58,
        marker=dict(colors=colors, line=dict(color=borders, width=2)),
        textinfo="label",
        textfont=dict(size=8, family=FONT, color="#cbd5e1"),
        hovertemplate="Digit %{label}: %{percent}<extra></extra>",
        direction="clockwise", rotation=0,
    ))
    fig.add_annotation(
        text=f"<b>{pred}</b>", x=0.5, y=0.54,
        font=dict(family=FONT, size=32, color=PURPLE), showarrow=False
    )
    fig.add_annotation(
        text=f"{probs[pred]*100:.1f}%", x=0.5, y=0.34,
        font=dict(family=FONT, size=9, color="#94a3b8"), showarrow=False
    )
    fig.update_layout(**base_layout(240), showlegend=False,
                      title=dict(text="BITSTREAM INTEGRITY", font=dict(size=8, color="#3b3b5c", family=FONT), x=0.5))
    return fig

def make_radar(probs):
    digits = [str(i) for i in range(10)]
    fig = go.Figure(go.Scatterpolar(
        r=list(probs) + [probs[0]],
        theta=digits + [digits[0]],
        fill='toself',
        fillcolor='rgba(168, 85, 247, 0.04)',
        line=dict(color=PURPLE, width=1.5),
        marker=dict(color=CYAN, size=4),
    ))
    fig.update_layout(
        **base_layout(240),
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(visible=True, showticklabels=False,
                            gridcolor=GRID, linecolor=GRID, range=[0, max(probs)+0.05]),
            angularaxis=dict(tickfont=dict(size=8, family=FONT, color="#4b4b74"),
                             gridcolor=GRID, linecolor=GRID),
        ),
        title=dict(text="FREQUENCY RADAR", font=dict(size=8, color="#3b3b5c", family=FONT), x=0.5),
    )
    return fig

def make_bar(probs, pred):
    colors = [f"rgba(168, 85, 247, {0.2 + 0.8*p/max(probs)})" for p in probs]
    colors[pred] = CYAN
    fig = go.Figure(go.Bar(
        x=[str(i) for i in range(10)], y=[p * 100 for p in probs],
        marker=dict(color=colors, line=dict(color=[CYAN if i == pred else "#221936" for i in range(10)], width=1.5)),
        hovertemplate="Digit %{x}: %{y:.1f}%<extra></extra>",
    ))
    fig.update_layout(
        **base_layout(240),
        xaxis=dict(gridcolor=GRID, linecolor=GRID, tickfont=dict(size=8, family=FONT, color="#4b4b74")),
        yaxis=dict(gridcolor=GRID, linecolor="#040209", tickfont=dict(size=7, family=FONT, color="#3b3b5c"),
                   ticksuffix="%", showgrid=True),
        title=dict(text="PROBABILITY ARRAY", font=dict(size=8, color="#3b3b5c", family=FONT), x=0.5),
        bargap=0.3,
    )
    return fig

def make_heatmap(probs):
    grid = np.array(probs).reshape(2, 5)
    labels = [["0","1","2","3","4"],["5","6","7","8","9"]]
    text   = [[f"{grid[r][c]*100:.1f}%" for c in range(5)] for r in range(2)]
    fig = go.Figure(go.Heatmap(
        z=grid, text=text, texttemplate="%{text}",
        colorscale=[[0, "#040209"], [0.4, "#090611"], [0.8, DARK_PURPLE], [1.0, CYAN]],
        showscale=False,
        xgap=4, ygap=4,
        textfont=dict(family=FONT, size=8, color="#cbd5e1"),
        customdata=labels,
        hovertemplate="Digit %{customdata}: %{z:.3f}<extra></extra>",
    ))
    fig.update_layout(
        **base_layout(160),
        xaxis=dict(tickvals=[0,1,2,3,4], ticktext=["0","1","2","3","4"],
                   tickfont=dict(family=FONT, size=8, color="#4b4b74"), showgrid=False),
        yaxis=dict(tickvals=[0,1], ticktext=["0–4","5–9"],
                   tickfont=dict(family=FONT, size=8, color="#4b4b74"), showgrid=False, autorange="reversed"),
        title=dict(text="DENSITY HEATMAP", font=dict(size=8, color="#3b3b5c", family=FONT), x=0.5),
    )
    return fig

# ── Dynamic Monospaced Square Matrix Background ──────────────────────────────
import streamlit.components.v1 as components
components.html("""
<script>
(function() {
    const doc = window.parent.document;

    const existing = doc.getElementById('pixel-bg');
    if (existing) existing.remove();

    if (!doc.getElementById('pixel-bg-style')) {
        const style = doc.createElement('style');
        style.id = 'pixel-bg-style';
        style.textContent = `
            #pixel-bg {
                position: fixed;
                top: 0; left: 0;
                width: 100vw; height: 100vh;
                pointer-events: none;
                z-index: 0;
            }
            [data-testid="stAppViewContainer"] > section,
            [data-testid="stHeader"] { position: relative; z-index: 1; }
        `;
        doc.head.appendChild(style);
    }

    const canvas = doc.createElement('canvas');
    canvas.id = 'pixel-bg';
    doc.body.appendChild(canvas);
    const ctx = canvas.getContext('2d');

    let squares = [];

    function randomBetween(a, b) { return a + Math.random() * (b - a); }

    function buildSquares() {
        squares = [];
        const W = canvas.width, H = canvas.height;
        const size = 12; 
        const cols = Math.ceil(W / 60);
        const rows = Math.ceil(H / 60);
        
        const COLORS = ['#a855f7', '#06b6d4', '#4c1d95'];
        
        for (let r = 0; r < rows; r++) {
            for (let c = 0; c < cols; c++) {
                if (Math.random() > 0.25) continue;
                const opacity = randomBetween(0.02, 0.12);
                const x = c * 60 + randomBetween(5, 45);
                const y = r * 60 + randomBetween(5, 45);
                const color = COLORS[Math.floor(Math.random() * COLORS.length)];
                squares.push({
                    x, y, size, color,
                    opacity,
                    speed: randomBetween(0.5, 1.5),
                    phase: randomBetween(0, Math.PI * 2),
                });
            }
        }
    }

    function resize() {
        canvas.width  = window.parent.innerWidth;
        canvas.height = window.parent.innerHeight;
        buildSquares();
    }

    function draw(ts) {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        const t = ts * 0.001;
        squares.forEach(sq => {
            const pulse = sq.opacity * (0.3 + 0.7 * Math.sin(t * sq.speed + sq.phase));
            ctx.globalAlpha = Math.max(0.01, pulse);
            ctx.fillStyle = sq.color;
            ctx.fillRect(sq.x, sq.y, sq.size, sq.size);
        });
        ctx.globalAlpha = 1;
        requestAnimationFrame(draw);
    }

    resize();
    window.parent.addEventListener('resize', resize);
    requestAnimationFrame(draw);
})();
</script>
""", height=0)


# ── Model ─────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('digit_recognition_model.h5')
model = load_model()

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-label">Quantized Neural Processing Unit</div>
    <h1 class="hero-title">pixeltalk</h1>
    <p class="hero-sub">Decoding structural raster inputs through an active tensor matrix.</p>
</div>
<div class="divider"></div>
""", unsafe_allow_html=True)

# ── Upload (centred) ──────────────────────────────────────────────────────────
_, mid_col, _ = st.columns([1, 2, 1])
with mid_col:
    uploaded_file = st.file_uploader(
        "Inject target 28x28 grayscale grid array",
        type=["png", "jpg", "jpeg"],
        label_visibility="visible",
    )

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# ── Main content ──────────────────────────────────────────────────────────────
if uploaded_file is not None:

    # Pre-process
    image         = Image.open(uploaded_file).convert('L')
    image_display = image.resize((140, 140), Image.NEAREST)
    image_model   = image.resize((28, 28))
    img_array     = np.array(image_model) / 255.0
    img_array     = img_array.reshape(1, 28, 28, 1)

    prediction      = model.predict(img_array, verbose=0)[0]
    predicted_digit = int(np.argmax(prediction))
    confidence      = float(prediction[predicted_digit]) * 100
    probs           = prediction.tolist()

    # ── 3-column layout ──────────────────────────────────────────────────────
    left, centre, right = st.columns([1.1, 1.6, 1.1], gap="large")

    # LEFT — donut + heatmap
    with left:
        st.markdown('<div class="chart-label">bitstream integrity</div>', unsafe_allow_html=True)
        st.plotly_chart(make_donut(probs, predicted_digit),
                        use_container_width=True, config={"displayModeBar": False})

        st.markdown('<div class="chart-label">density heatmap</div>', unsafe_allow_html=True)
        st.plotly_chart(make_heatmap(probs),
                        use_container_width=True, config={"displayModeBar": False})

    # CENTRE — image + result card + prob bars
    with centre:
        col_img, col_res = st.columns([1, 1.5], gap="medium")
        with col_img:
            st.markdown('<div class="section-title">Raster Input</div>', unsafe_allow_html=True)
            st.image(image_display, width=130)
        with col_res:
            st.markdown(f"""
            <div class="result-card">
                <div class="result-label">Output Node</div>
                <div class="result-digit">{predicted_digit}</div>
                <div class="result-confidence">// {confidence:.2f}% accuracy yield</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Activation Signatures</div>', unsafe_allow_html=True)

        bars_html = ""
        for i in np.argsort(prediction)[::-1]:
            d   = int(i)
            pct = float(prediction[d]) * 100
            bars_html += f"""
            <div class="prob-row">
                <span class="prob-digit-label">{d}</span>
                <div class="prob-bar-bg"><div class="prob-bar-fill" style="width:{pct:.1f}%"></div></div>
                <span class="prob-pct">{pct:.1f}%</span>
            </div>"""
        st.markdown(bars_html, unsafe_allow_html=True)

    # RIGHT — bar + radar
    with right:
        st.markdown('<div class="chart-label">probability array</div>', unsafe_allow_html=True)
        st.plotly_chart(make_bar(probs, predicted_digit),
                        use_container_width=True, config={"displayModeBar": False})

        st.markdown('<div class="chart-label">frequency radar</div>', unsafe_allow_html=True)
        st.plotly_chart(make_radar(probs),
                        use_container_width=True, config={"displayModeBar": False})

else:
    st.markdown("""
    <div class="placeholder-box">
        SYSTEM IDLE // Awaiting matrix array injection...
    </div>
    """, unsafe_allow_html=True)

st.markdown('<div class="footer">PixelTalk // Feedforward Convolutional MNIST Engine // Ver 2.0.46</div>', unsafe_allow_html=True)