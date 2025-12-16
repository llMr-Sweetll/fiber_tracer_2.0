import os
import sys
import time
import subprocess
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, Confirm

def print_welcome():
    console = Console()
    console.print(Panel.fit(
        "[bold cyan]FIBER TRACER 2.0 - WIZARD[/bold cyan]\n"
        "[yellow]Official Algorithm by Chandrashekhar Hegde[/yellow]\n\n"
        "This tool will guide you through analyzing your fiber data.\n"
        "No coding required.",
        title="🚀 To Infinity and Beyond",
        border_style="magenta"
    ))
    time.sleep(1)

def main():
    try:
        from rich.console import Console
    except ImportError:
        print("Installing wizard dependencies...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "rich"])
        from rich.console import Console

    console = Console()
    print_welcome()

    # Step 1: Input Data
    console.print("\n[bold green]STEP 1: Locate your Data[/bold green]")
    default_dir = os.path.join(os.getcwd(), "sample_data")
    data_dir = Prompt.ask("Enter the folder path containing your TIFF images", default=default_dir)

    while not os.path.exists(data_dir):
        console.print(f"[red]Error:[/red] The path '{data_dir}' does not exist.")
        data_dir = Prompt.ask("Please enter a valid folder path")

    # Step 2: Output Data
    console.print("\n[bold green]STEP 2: Where to save results?[/bold green]")
    default_out = os.path.join(os.getcwd(), "results")
    output_dir = Prompt.ask("Enter the output folder path", default=default_out)

    # Step 3: Parameters (Simplified)
    console.print("\n[bold green]STEP 3: Analysis Settings[/bold green]")
    voxel_size = Prompt.ask("What is your voxel size? (micrometers/pixel)", default="1.1")
    
    use_ai_killer = Confirm.ask("Enable HT3 'AI-Killer' Algorithms? (Recommended for precision)", default=True)
    
    # Construct Command
    cmd = [
        sys.executable, "fiber_tracer_v2.py",
        "--data_dir", data_dir,
        "--output_dir", output_dir,
        "--voxel_size", voxel_size,
        "--log_level", "INFO"
    ]

    if use_ai_killer:
        # Implicitly enabled by default in v2 code, but we can be explicit if flags exist
        # config.analysis.use_structure_tensor is True by default in our code
        pass 

    console.print("\n[bold yellow]Ready to Launch![/bold yellow]")
    console.print(f"Command: {' '.join(cmd)}")
    
    if Confirm.ask("Start Analysis?", default=True):
        console.print("\n[bold magenta]Running Fiber Tracer... (Please wait)[/bold magenta]")
        start_time = time.time()
        
        process = subprocess.Popen(cmd)
        process.wait()
        
        if process.returncode == 0:
            console.print(Panel(
                f"[bold green]SUCCESS![/bold green]\n\n"
                f"Analysis complete in {time.time() - start_time:.1f} seconds.\n"
                f"Results saved to: [underline]{output_dir}[/underline]\n\n"
                "Type: [bold cyan]start " + os.path.join(output_dir, "premium_report.html") + "[/bold cyan] to view your dashboard.",
                title="Mission Accomplished 🏁",
                border_style="green"
            ))
            
            if Confirm.ask("Open Report now?", default=True):
                if os.name == 'nt':
                    os.startfile(os.path.join(output_dir, "premium_report.html"))
        else:
            console.print("[bold red]Analysis Failed.[/bold red] Check the logs above.")

if __name__ == "__main__":
    main()
