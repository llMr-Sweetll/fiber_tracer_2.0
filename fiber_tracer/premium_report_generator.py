"""
Premium Report Generator
Generates high-end HTML reports for Fiber Tracer analysis.

Author: Chandrashekhar Hegde
"""

import os
import datetime
from typing import Dict, Any, List
import json

class PremiumReportGenerator:
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.css = """
        :root {
            --primary: #6366f1;
            --surface: #1e293b;
            --background: #0f172a;
            --text: #f8fafc;
            --success: #10b981;
            --warning: #f59e0b;
        }
        
        body {
            font-family: 'Inter', sans-serif;
            background-color: var(--background);
            color: var(--text);
            margin: 0;
            padding: 2rem;
            line-height: 1.6;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
        }
        
        .header {
            background: rgba(30, 41, 59, 0.7);
            backdrop-filter: blur(10px);
            border-radius: 1rem;
            padding: 2rem;
            margin-bottom: 2rem;
            border: 1px solid rgba(255, 255, 255, 0.1);
            text-align: center;
        }
        
        .title {
            font-size: 2.5rem;
            font-weight: 700;
            margin: 0;
            background: linear-gradient(to right, #818cf8, #c084fc);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        
        .meta {
            color: #94a3b8;
            margin-top: 0.5rem;
        }
        
        .dashboard-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 1.5rem;
            margin-bottom: 2rem;
        }
        
        .card {
            background: var(--surface);
            border-radius: 1rem;
            padding: 1.5rem;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
            border: 1px solid rgba(255, 255, 255, 0.05);
            transition: transform 0.2s;
        }
        
        .card:hover {
            transform: translateY(-5px);
        }
        
        .metric {
            font-size: 2rem;
            font-weight: 700;
            color: var(--primary);
        }
        
        .label {
            font-size: 0.875rem;
            color: #94a3b8;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        
        .chip-container {
            display: flex;
            gap: 0.5rem;
            flex-wrap: wrap;
            margin-top: 1rem;
        }
        
        .chip {
            display: inline-flex;
            align-items: center;
            padding: 0.25rem 0.75rem;
            border-radius: 9999px;
            font-size: 0.75rem;
            font-weight: 600;
            background: rgba(99, 102, 241, 0.2);
            color: #818cf8;
            border: 1px solid rgba(99, 102, 241, 0.3);
        }
        
        .chip.success { background: rgba(16, 185, 129, 0.2); color: #34d399; border-color: rgba(16, 185, 129, 0.3); }
        .chip.warning { background: rgba(245, 158, 11, 0.2); color: #fbbf24; border-color: rgba(245, 158, 11, 0.3); }
        
        table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 1rem;
        }
        
        th, td {
            text-align: left;
            padding: 1rem;
            border-bottom: 1px solid rgba(255,255,255,0.1);
        }
        
        th {
            color: #94a3b8;
            font-weight: 600;
        }
        
        .iframe-container {
            width: 100%;
            height: 600px;
            border: none;
            border-radius: 1rem;
            overflow: hidden;
            margin-top: 2rem;
        }
        """

    def _generate_plot_html(self, fibers: List[Dict[str, Any]]) -> str:
        """Generates the HTML string for the interactive 3D plot."""
        try:
            import plotly.graph_objects as go
            import pandas as pd
            
            if not fibers:
                return "<div style='text-align:center; padding: 2rem; color: #94a3b8;'>No fiber data available for visualization</div>"
                
            df = pd.DataFrame(fibers)
            
            # Check required columns
            required_cols = ['Centroid X (μm)', 'Centroid Y (μm)', 'Centroid Z (μm)', 'Length (μm)']
            if not all(col in df.columns for col in required_cols):
                 return "<div style='text-align:center; padding: 2rem; color: #f59e0b;'>Missing data fields for 3D visualization</div>"

            fig = go.Figure(data=[go.Scatter3d(
                x=df['Centroid X (μm)'],
                y=df['Centroid Y (μm)'],
                z=df['Centroid Z (μm)'],
                mode='markers',
                marker=dict(
                    size=3,
                    color=df['Length (μm)'],
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Length (μm)"),
                    opacity=0.8
                ),
                text=[f"ID: {row.get('Fiber ID', 'N/A')}<br>Length: {row.get('Length (μm)', 0):.1f}" for _, row in df.iterrows()],
                hoverinfo='text'
            )])
            
            fig.update_layout(
                title='Interactive 3D Fiber Network',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#f8fafc', family="Inter"),
                scene=dict(
                    xaxis=dict(title='X (μm)', gridcolor='rgba(255,255,255,0.1)', showbackground=False),
                    yaxis=dict(title='Y (μm)', gridcolor='rgba(255,255,255,0.1)', showbackground=False),
                    zaxis=dict(title='Z (μm)', gridcolor='rgba(255,255,255,0.1)', showbackground=False),
                ),
                margin=dict(l=0, r=0, b=0, t=40),
                height=500
            )
            
            return fig.to_html(full_html=False, include_plotlyjs='cdn')
            
        except ImportError:
            return "<div style='text-align:center; padding: 2rem; color: #f59e0b;'>Plotly not installed. Visualization disabled.</div>"
        except Exception as e:
            return f"<div style='text-align:center; padding: 2rem; color: #ef4444;'>Error generating visualization: {str(e)}</div>"

    def generate(self, stats: Dict[str, Any], fibers: List[Dict[str, Any]] = None):
        """Generates the HTML report."""
        
        # Calculate derived metrics for Chips
        alignment_score = stats.get('orientation_stats', {}).get('mean', 0)
        alignment_class = "success" if alignment_score < 15 else "warning"
        alignment_text = "Highly Aligned" if alignment_score < 15 else "Misaligned"
        
        vf = stats.get('volume_fraction', 0)
        density_text = "High Density" if vf > 10 else "Low Density"
        
        # Generate 3D Plot
        plot_html = self._generate_plot_html(fibers)
        
        html = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Fiber Analysis Report</title>
            <style>{self.css}</style>
            <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap" rel="stylesheet">
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        </head>
        <body>
            <div class="container">
                <header class="header">
                    <h1 class="title">Fiber Analysis Report</h1>
                    <div class="meta">Generated by Fiber Tracer v2.0 • Author: Chandrashekhar Hegde • {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}</div>
                    
                    <div class="chip-container" style="justify-content: center; margin-top: 1rem;">
                        <span class="chip {alignment_class}">Status: {alignment_text}</span>
                        <span class="chip">Algorithm: HT3 (Structure Tensor)</span>
                        <span class="chip">{density_text}</span>
                    </div>
                </header>
                
                <div class="dashboard-grid">
                    <div class="card">
                        <div class="label">Total Fibers</div>
                        <div class="metric">{stats.get('total_fibers', 0):,}</div>
                    </div>
                    
                    <div class="card">
                        <div class="label">Volume Fraction</div>
                        <div class="metric">{stats.get('volume_fraction', 0):.2f}%</div>
                    </div>
                    
                    <div class="card">
                        <div class="label">Avg Diameter</div>
                        <div class="metric">{stats.get('diameter_stats', {}).get('mean', 0):.2f} μm</div>
                        <div class="chip-container">
                            <span class="chip">σ = {stats.get('diameter_stats', {}).get('std', 0):.2f}</span>
                        </div>
                    </div>
                    
                    <div class="card">
                        <div class="label">Avg Orientation</div>
                        <div class="metric">{stats.get('orientation_stats', {}).get('mean', 0):.2f}°</div>
                    </div>
                </div>
                
                <div class="card">
                    <h2 style="margin-top: 0;">Interactive Visualization</h2>
                    <p style="color: #94a3b8;">3D Reconstruction of Fiber Network using HT3 Algorithm</p>
                    <div class="iframe-container" style="background: rgba(0,0,0,0.2);">
                        {plot_html}
                    </div>
                </div>
                
                <div class="card" style="margin-top: 2rem;">
                    <h2 style="margin-top: 0;">Quantitative Breakdown</h2>
                    <table>
                        <thead>
                            <tr>
                                <th>Metric</th>
                                <th>Mean</th>
                                <th>Std Dev</th>
                                <th>Min</th>
                                <th>Max</th>
                            </tr>
                        </thead>
                        <tbody>
                            <tr>
                                <td>Length (μm)</td>
                                <td>{stats.get('length_stats', {}).get('mean', 0):.2f}</td>
                                <td>{stats.get('length_stats', {}).get('std', 0):.2f}</td>
                                <td>{stats.get('length_stats', {}).get('min', 0):.2f}</td>
                                <td>{stats.get('length_stats', {}).get('max', 0):.2f}</td>
                            </tr>
                            <tr>
                                <td>Diameter (μm)</td>
                                <td>{stats.get('diameter_stats', {}).get('mean', 0):.2f}</td>
                                <td>{stats.get('diameter_stats', {}).get('std', 0):.2f}</td>
                                <td>{stats.get('diameter_stats', {}).get('min', 0):.2f}</td>
                                <td>{stats.get('diameter_stats', {}).get('max', 0):.2f}</td>
                            </tr>
                            <tr>
                                <td>Orientation (°)</td>
                                <td>{stats.get('orientation_stats', {}).get('mean', 0):.2f}</td>
                                <td>{stats.get('orientation_stats', {}).get('std', 0):.2f}</td>
                                <td>{stats.get('orientation_stats', {}).get('min', 0):.2f}</td>
                                <td>{stats.get('orientation_stats', {}).get('max', 0):.2f}</td>
                            </tr>
                             <tr>
                                <td>Tortuosity</td>
                                <td>{stats.get('tortuosity_stats', {}).get('mean', 0):.2f}</td>
                                <td>{stats.get('tortuosity_stats', {}).get('std', 0):.2f}</td>
                                <td>{stats.get('tortuosity_stats', {}).get('min', 0):.2f}</td>
                                <td>{stats.get('tortuosity_stats', {}).get('max', 0):.2f}</td>
                            </tr>
                        </tbody>
                    </table>
                </div>
            </div>
        </body>
        </html>
        """
        
        output_path = os.path.join(self.output_dir, "premium_report.html")
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html)
        
        print(f"Premium report generated at: {output_path}")

# Example Usage Stub
if __name__ == "__main__":
    import random
    
    # Mock stats
    mock_stats = {
        'total_fibers': 150,
        'volume_fraction': 12.5,
        'diameter_stats': {'mean': 5.2, 'std': 1.1, 'min': 2.0, 'max': 8.5},
        'length_stats': {'mean': 150.4, 'std': 45.2, 'min': 20.0, 'max': 300.0},
        'orientation_stats': {'mean': 12.4, 'std': 5.6, 'min': 0.0, 'max': 88.0},
        'tortuosity_stats': {'mean': 1.05, 'std': 0.02, 'min': 1.0, 'max': 1.2}
    }
    
    # Generate mock fiber data for visualization
    mock_fibers = []
    for i in range(150):
        mock_fibers.append({
            'Fiber ID': i,
            'Centroid X (μm)': random.uniform(0, 100),
            'Centroid Y (μm)': random.uniform(0, 100),
            'Centroid Z (μm)': random.uniform(0, 100),
            'Length (μm)': random.uniform(50, 200),
        })
        
    generator = PremiumReportGenerator(".")
    generator.generate(mock_stats, mock_fibers)

