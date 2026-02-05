"""
Rhino Multi-View Screenshot Generator
======================================
Run in Rhino: _RunPythonScript "C:\Users\vivek\Desktop\code space\RAG\scripts\rhino_screenshots.py"
"""

import rhinoscriptsyntax as rs
import Rhino
import os

# Configuration
BASE_DIR = r"C:\Users\vivek\Desktop\code space\RAG"
PRONGS_IN = os.path.join(BASE_DIR, "cad_library", "prongs")
STONES_IN = os.path.join(BASE_DIR, "cad_library", "stones")
PRONGS_OUT = os.path.join(BASE_DIR, "prongs_sc")
STONES_OUT = os.path.join(BASE_DIR, "stones_sc")
SIZE = 800

def capture_views(output_dir, base_name):
    """Capture 4 views of current model"""
    
    views = [
        ("Perspective", "Perspective"),
        ("Front", "Front"),
        ("Right", "Right"),
        ("Top", "Top")
    ]
    
    for view_name, file_suffix in views:
        # Set the view
        rs.Command("-_SetView World {} _Enter".format(view_name), False)
        
        # Set shaded mode
        rs.Command("-_SetDisplayMode Mode=Shaded _Enter", False)
        
        # Zoom extents
        rs.Command("-_Zoom All Extents _Enter", False)
        
        # Wait for redraw
        rs.Sleep(100)
        Rhino.RhinoApp.Wait()
        
        # Capture
        out_path = os.path.join(output_dir, "{}_{}.png".format(base_name, file_suffix.lower()))
        rs.Command('-_ViewCaptureToFile "{}" Width={} Height={} Scale=1 DrawGrid=No DrawWorldAxes=No DrawCPlaneAxes=No TransparentBackground=No _Enter'.format(
            out_path, SIZE, SIZE), False)
    
    # Check if files were created
    test_file = os.path.join(output_dir, "{}_perspective.png".format(base_name))
    return os.path.exists(test_file)

def process_folder(input_dir, output_dir, name):
    """Process all .3dm files in a folder"""
    
    if not os.path.exists(input_dir):
        print("ERROR: {} not found".format(input_dir))
        return 0
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    files = [f for f in os.listdir(input_dir) if f.lower().endswith('.3dm')]
    total = len(files)
    success = 0
    
    print("\n{}: {} files".format(name, total))
    
    for i, fname in enumerate(files):
        base_name = os.path.splitext(fname)[0]
        
        # Skip if already done
        check = os.path.join(output_dir, "{}_perspective.png".format(base_name))
        if os.path.exists(check):
            print("[{}/{}] {} - skipped".format(i+1, total, fname))
            success += 1
            continue
        
        # Open file - discard changes from previous file, no save prompt
        fpath = os.path.join(input_dir, fname)
        rs.DocumentModified(False)  # Mark current doc as not modified
        Rhino.RhinoDoc.Open(fpath)
        Rhino.RhinoApp.Wait()
        rs.Sleep(200)
        
        # Check if objects loaded
        if not rs.AllObjects():
            print("[{}/{}] {} - no objects".format(i+1, total, fname))
            continue
        
        # Capture views
        if capture_views(output_dir, base_name):
            print("[{}/{}] {} - done".format(i+1, total, fname))
            success += 1
        else:
            print("[{}/{}] {} - FAILED".format(i+1, total, fname))
    
    return success

def main():
    choice = rs.ListBox(
        ["Test 1 prong", "Test 1 stone", "All prongs", "All stones", "Everything"],
        "What to process?",
        "Screenshot Generator"
    )
    
    if not choice:
        return
    
    if choice == "Test 1 prong":
        files = [f for f in os.listdir(PRONGS_IN) if f.endswith('.3dm')]
        if files:
            fpath = os.path.join(PRONGS_IN, files[0])
            rs.Command('-_Open "{}" _Enter'.format(fpath), False)
            Rhino.RhinoApp.Wait()
            if not os.path.exists(PRONGS_OUT):
                os.makedirs(PRONGS_OUT)
            base = os.path.splitext(files[0])[0]
            if capture_views(PRONGS_OUT, base):
                rs.MessageBox("Success! Check:\n{}".format(PRONGS_OUT))
            else:
                rs.MessageBox("Failed to capture")
    
    elif choice == "Test 1 stone":
        files = [f for f in os.listdir(STONES_IN) if f.endswith('.3dm')]
        if files:
            fpath = os.path.join(STONES_IN, files[0])
            rs.Command('-_Open "{}" _Enter'.format(fpath), False)
            Rhino.RhinoApp.Wait()
            if not os.path.exists(STONES_OUT):
                os.makedirs(STONES_OUT)
            base = os.path.splitext(files[0])[0]
            if capture_views(STONES_OUT, base):
                rs.MessageBox("Success! Check:\n{}".format(STONES_OUT))
            else:
                rs.MessageBox("Failed to capture")
    
    elif choice == "All prongs":
        n = process_folder(PRONGS_IN, PRONGS_OUT, "Prongs")
        rs.MessageBox("Done! {} prongs processed".format(n))
    
    elif choice == "All stones":
        n = process_folder(STONES_IN, STONES_OUT, "Stones")
        rs.MessageBox("Done! {} stones processed".format(n))
    
    elif choice == "Everything":
        n1 = process_folder(PRONGS_IN, PRONGS_OUT, "Prongs")
        n2 = process_folder(STONES_IN, STONES_OUT, "Stones")
        rs.MessageBox("Done!\nProngs: {}\nStones: {}".format(n1, n2))

if __name__ == "__main__":
    main()
