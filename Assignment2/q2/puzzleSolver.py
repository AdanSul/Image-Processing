import cv2
import numpy as np
import os
import shutil
import sys

#matches is of (3|4 X 2 X 2) size. Each row is a match - pair of (kp1,kp2) where kpi = (x,y)
def get_transform(matches, is_affine):
    src_points, dst_points = matches[:, 0], matches[:, 1]
    
    # Convert points to float32
    src_points = src_points.astype(np.float32)
    dst_points = dst_points.astype(np.float32)
    
    # Calculate the transformation matrix
    if is_affine:
        T = cv2.getAffineTransform(src_points, dst_points)
    else:
        T = cv2.getPerspectiveTransform(src_points, dst_points)
    
    return T


def stitch(img1, img2):
    # Perform weighted blending between the base image and overlay image
    blended = cv2.addWeighted(img1, 0.5, img2, 0.5, 0)
    
    # Highlight differences to smooth transitions
    difference = cv2.absdiff(img1, img2)
    final_stitched = cv2.addWeighted(blended, 1.0, difference, 0.5, 0)
    
    return final_stitched


# Output size is (w,h)
def inverse_transform_target_image(target_img, original_transform, output_size):
    if original_transform.shape == (3, 3):  # Projective transformation
        inverse_transform_matrix = np.linalg.inv(original_transform)
        return cv2.warpPerspective(target_img, inverse_transform_matrix, output_size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_TRANSPARENT)
    elif original_transform.shape == (2, 3):  # Affine transformation
        inverse_transform_matrix = cv2.invertAffineTransform(original_transform)
        return cv2.warpAffine(target_img, inverse_transform_matrix, output_size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_TRANSPARENT)


# returns list of pieces file names
def prepare_puzzle(puzzle_dir):
	edited = os.path.join(puzzle_dir, 'abs_pieces')
	if os.path.exists(edited):
		shutil.rmtree(edited)
	os.mkdir(edited)
	
	affine = 4 - int("affine" in puzzle_dir)
	
	matches_data = os.path.join(puzzle_dir, 'matches.txt')
	n_images = len(os.listdir(os.path.join(puzzle_dir, 'pieces')))

	matches = np.loadtxt(matches_data, dtype=np.int64).reshape(n_images-1,affine,2,2)
	
	return matches, affine == 3, n_images

def get_image_size(image):
    height, width = image.shape[:2]
    return (width, height)

if __name__ == '__main__':
    lst = ['puzzle_affine_1', 'puzzle_affine_2', 'puzzle_homography_1']

    for puzzle_dir in lst:
        print(f'Starting {puzzle_dir}')
        
        puzzle = os.path.join('puzzles', puzzle_dir)
        pieces_pth = os.path.join(puzzle, 'pieces')
        edited = os.path.join(puzzle, 'abs_pieces')
        
        matches, is_affine, n_images = prepare_puzzle(puzzle)

        # Load the first piece as the base canvas
        piece1_path = os.path.join(pieces_pth, 'piece_1.jpg')
        piece1 = cv2.imread(piece1_path)

        # Save the first piece
        abs_piece_path = os.path.join(edited, 'piece_1_absolute.jpg')
        cv2.imwrite(abs_piece_path, piece1)

        # Iterate through the remaining pieces
        for i in range(2, n_images + 1):
            # Load the next piece
            piecex_path = os.path.join(pieces_pth, f'piece_{i}.jpg')
            piecex = cv2.imread(piecex_path)
            
            # Get the transformation matrix
            transform_matrix = get_transform(matches[i - 2], is_affine)
            
            # Perform inverse transformation on the target image
            inverse_transformed_piece = inverse_transform_target_image(piecex, transform_matrix, get_image_size(piece1))
            
            # Save the transformed piece
            abs_piece_path = os.path.join(edited, f'piece_{i}_absolute.jpg')
            cv2.imwrite(abs_piece_path, inverse_transformed_piece)

            # # Display the transformed piece
            # cv2.imshow(f"Inverse Transformed Piece {i}", inverse_transformed_piece)
            # cv2.waitKey(0)

            # Stitch the transformed piece with the canvas
            piece1 = stitch(piece1, inverse_transformed_piece)
            final_puzzle = piece1
            
            # # Display the stitched result
            # cv2.imshow(f"Stitched Image 1-{i}", piece1)
            # cv2.waitKey(0)

        # Save the final stitched image as the solution
        sol_file = f'solution.jpg' 
        cv2.imwrite(os.path.join(puzzle, sol_file), final_puzzle)
        print(f"Final solution saved as {sol_file}")

        # Close all OpenCV windows
        cv2.destroyAllWindows()
