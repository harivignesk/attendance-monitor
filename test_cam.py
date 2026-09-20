import cv2

print("Testing camera backends...")

def test_camera():
    backends = [
        ("CAP_DSHOW (DirectShow)", cv2.CAP_DSHOW),
        ("CAP_MSMF (Media Foundation)", cv2.CAP_MSMF),
        ("DEFAULT", cv2.CAP_ANY)
    ]

    for name, backend in backends:
        print(f"\nTrying backend: {name}")
        for idx in range(3):
            cap = cv2.VideoCapture(idx, backend)
            if cap.isOpened():
                ret, frame = cap.read()
                cap.release()
                if ret and frame is not None:
                    print(f"  SUCCESS! Index {idx} with {name} works! Frame shape: {frame.shape}")
                    return idx, backend
                else:
                    print(f"  Index {idx} opened but failed to read frame.")
            else:
                print(f"  Index {idx} could not be opened.")

    print("\nNo working camera device found.")
    return None, None

if __name__ == "__main__":
    idx, backend = test_camera()
