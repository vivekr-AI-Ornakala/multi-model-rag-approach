
import rhinoscriptsyntax as rs
def assemble():
    rs.EnableRedraw(False)
    shank_z = 0
    if r"": 
        rs.Command('_-Import "' + r"" + '" _Enter')
        shanks = rs.LastCreatedObjects()
        if shanks: shank_z = rs.BoundingBox(shanks)[4].Z
    if r"C:\\Users\\vivek\\Desktop\\code space\\RAG\\cad_library\\Prongs\\391_391_1202_S.3dm": 
        rs.Command('_-Import "' + r"C:\\Users\\vivek\\Desktop\\code space\\RAG\\cad_library\\Prongs\\391_391_1202_S.3dm" + '" _Enter')
        h = rs.LastCreatedObjects()
        if h: 
            box = rs.BoundingBox(h)
            ctr = (box[0] + box[6]) / 2
            rs.MoveObjects(h, [0-ctr.X, 0-ctr.Y, 0-ctr.Z])
            rs.MoveObjects(h, [0, 0, shank_z])
    if r"C:\\Users\\vivek\\Desktop\\code space\\RAG\\cad_library\\Stones\\C 5 MM.3dm":
        rs.Command('_-Import "' + r"C:\\Users\\vivek\\Desktop\\code space\\RAG\\cad_library\\Stones\\C 5 MM.3dm" + '" _Enter')
        s = rs.LastCreatedObjects()
        if s: 
            box = rs.BoundingBox(s)
            ctr = (box[0] + box[6]) / 2
            rs.MoveObjects(s, [0-ctr.X, 0-ctr.Y, 0-ctr.Z])
            rs.MoveObjects(s, [0, 0, shank_z + 1.0])
    rs.EnableRedraw(True)
    print("Smart Assembly Done")
if __name__=="__main__": assemble()
