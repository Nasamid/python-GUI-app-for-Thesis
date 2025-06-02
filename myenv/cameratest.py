import argparse
from rembg import remove
from PIL import Image
import io

def main():
    parser = argparse.ArgumentParser(description='Remove image background using rembg')
    parser.add_argument('input', type=str, help='Input image file path')
    parser.add_argument('output', type=str, help='Output image file path (PNG format recommended for transparency)')
    args = parser.parse_args()

    # Read the input image in binary mode
    with open(args.input, 'rb') as i:
        input_data = i.read()

    # Remove the background
    result = remove(input_data)

    # Write the result to the output file
    with open(args.output, 'wb') as o:
        o.write(result)

    print(f"Background removed. Output saved as {args.output}")

if __name__ == '__main__':
    main()