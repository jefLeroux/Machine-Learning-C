#define MLIGHT_IMPLEMENTATION
#include "MLight.h"

#define OLIVEC_IMPLEMENTATION
#include "thirdparty/olive.c"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "thirdparty/stb_image_write.h"

#define IMG_FACTOR 100
#define IMG_WIDTH (16*IMG_FACTOR)
#define IMG_HEIGHT (9*IMG_FACTOR)

uint32_t img_pixels[IMG_WIDTH * IMG_HEIGHT];

void nn_render(Olivec_Canvas img, NN nn) {
    uint32_t background_color = 0xFF181818;
    uint32_t low_color = 0x000000FF;
    uint32_t high_color = 0x00FFA500;
    olivec_fill(img, background_color);

    int neuron_radius = 25;
    int layer_border_vpad = 50;
    int layer_border_hpad = 50;
    int nn_width = img.width - 2*layer_border_hpad;
    int nn_height = img.height - 2*layer_border_vpad;
    int nn_x = img.width/2 - nn_width/2;
    int nn_y = img.height/ 2 - nn_height/2;
    size_t arch_count = nn.count + 1;
    int layer_hpad = nn_width / arch_count;
    for(size_t l = 0; l < nn.as[0].cols; l++) {
        int layer_vpad1 = nn_height / nn.as[l].cols;
        for (size_t i = 0; i < nn.as[l].cols; ++i) {
            int cx1 = nn_x + l*layer_hpad + layer_hpad/2;
            int cy1 = nn_y + i*layer_vpad1  + layer_vpad1/2;
            if (l < arch_count - 1) {
                int layer_vpad2 = nn_height / nn.as[l + 1].cols;
                for (size_t j = 0; j < nn.as[l+1].cols; ++j) {
                    int cx2 = nn_x + (l+1)*layer_hpad + layer_hpad/2; 
                    int cy2 = nn_y + j*layer_vpad2  + layer_vpad2/2;

                    uint32_t alpha = floorf(255.0f * sigmoidf(MATRIX_AT(nn.ws[l], j, i)));
                    uint32_t connection_color = 0xFF000000|low_color;
                    olivec_blend_color(&connection_color, (alpha<<(8 * 3))|high_color);
                    olivec_line(img, cx1, cy1, cx2, cy2, connection_color);
                }
            }
            if (l > 0) {
                uint32_t alpha = floorf(255.0f * sigmoidf(MATRIX_AT(nn.bs[l-1], 0, i)));
                uint32_t neuron_color = 0xFF000000|low_color;
                olivec_blend_color(&neuron_color, (alpha<<(8 * 3))|high_color);
                olivec_circle(img, cx1, cy1, neuron_radius, neuron_color);
            } else {
                olivec_circle(img, cx1, cy1, neuron_radius, 0xFFAAAAAA);
            }
        }
    }
}

int main(void) {
    srand(time(0));

    size_t arch[] = {4, 4, 2, 1};
    size_t arch_count = ARRAY_LEN(arch);
    NN nn = nn_alloc(arch, arch_count);
    nn_randomize(nn, -10, 10);
    
    Olivec_Canvas img = olivec_canvas(img_pixels, IMG_WIDTH, IMG_HEIGHT, IMG_WIDTH);
    nn_render(img, nn);

    uint32_t frame_thic = 10;
    uint32_t frame_color = 0xFF252525;
    olivec_frame(img, 0, 0, img.width-1, img.height-1, frame_thic, frame_color);

    const char *img_file_path = "/out/nn.png";
    if(!stbi_write_png(img_file_path, img.width, img.height, 4, img_pixels, img.stride*sizeof(uint32_t))) {
        printf("ERORR: stbi_write_png(%s) failed\n", img_file_path);
        return 1;
    }
    printf("Saved image to %s\n", img_file_path);

    return 0;
}