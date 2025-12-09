#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <memory>
#include <assert.h>
#include <string>

class MNISTDataset final {
public:
    MNISTDataset()
        : m_count(0),
        m_width(0),
        m_height(0),
        m_imageSize(0),
        m_buffer(nullptr),
        m_imageBuffer(nullptr),
        m_categoryBuffer(nullptr)
    {
    }

    ~MNISTDataset() {
        if (m_buffer) free(m_buffer);
        if (m_categoryBuffer) free(m_categoryBuffer);
    }

    void Print() {
        for (int n = 0; n < m_count; ++n) {
            const float* imageBuffer = &m_imageBuffer[n * m_imageSize];
            for (int j = 0; j < m_height; ++j) {
                for (int i = 0; i < m_width; ++i) {
                    printf("%3d ", (uint8_t)imageBuffer[j * m_width + i]);
                }
                printf("\n");
            }

            printf("\n [%u] ===> cat(%u)\n\n", n, m_categoryBuffer[n]);
        }
    }

    int GetImageWidth() const {
        return m_width;
    }

    int GetImageHeight() const {
        return m_height;
    }

    int GetImageCount() const {
        return m_count;
    }

    int GetImageSize() const {
        return m_imageSize;
    }

    const float* GetImageData() const {
        return m_imageBuffer;
    }

    const uint8_t* GetCategoryData() const {
        return m_categoryBuffer;
    }

    int Parse(const char* imageFile, const char* labelFile, bool verbose) {
        FILE* fimg = fopen(imageFile, "rb");
        if (!fimg) {
            printf("Failed to open %s for reading\n", imageFile);
            return 1;
        }

        FILE* flabel = fopen(labelFile, "rb");
        if (!flabel) {
            printf("Failed to open %s for reading\n", labelFile);
            return 1;
        }
        std::shared_ptr<void> autofimg(nullptr, [fimg, flabel](void*) {
            if (fimg) fclose(fimg);
            if (flabel) fclose(flabel);
        });

        uint32_t value;

        // Read magic number
        assert(!feof(fimg));
        if (fread(&value, sizeof(uint32_t), 1, fimg) != 1) {
            printf("Failed to read magic number from %s", imageFile);
            return 1;
        }

        assert(__builtin_bswap32(value) == 0x00000803);
        if (verbose)
            printf("Image Magic        :%0X%u\n",
                __builtin_bswap32(value), __builtin_bswap32(value));

        // Read count
        assert(!feof(fimg));
        if (fread(&value, sizeof(uint32_t), 1, fimg) != 1) {
            printf("Failed to read image count from %s", imageFile);
            return 1;
        }
        const uint32_t count = __builtin_bswap32(value);
        assert(count > 0);
        if (verbose)
            printf("Image Count        :%0X%u\n", count, count);

        // Read rows
        assert(!feof(fimg));
        if (fread(&value, sizeof(uint32_t), 1, fimg) != 1) {
            printf("Failed to read number of rows from %s", imageFile);
            return 1;
        }
        const uint32_t rows = __builtin_bswap32(value);
        assert(rows > 0);
        if (verbose)
            printf("Image Rows         :%0X%u\n", rows, rows);

        // Read cols
        assert(!feof(fimg));
        if (fread(&value, sizeof(uint32_t), 1, fimg) != 1) {
            printf("Failed to read number of columns from %s", imageFile);
        }
        const uint32_t cols = __builtin_bswap32(value);
        assert(cols > 0);
        if (verbose)
            printf("Image Columns      :%0X%u\n", cols, cols);

        // Read magic number (label)
        assert(!feof(flabel));
        if (fread(&value, sizeof(uint32_t), 1, flabel) != 1) {
            printf("Failed to read magic number from %s", labelFile);
            return 1;
        }
        assert(__builtin_bswap32(value) == 0x00000801);
        if (verbose)
            printf("Label Magic        :%0X%u\n",
                __builtin_bswap32(value), __builtin_bswap32(value));

        // Read label count
        assert(!feof(flabel));
        if (fread(&value, sizeof(uint32_t), 1, flabel) != 1) {
            printf("Failed to read label count from %s", labelFile);
            return 1;
        }
        assert(__builtin_bswap32(value) == count);
        if (verbose)
            printf("Label Count        :%0X%u\n",
                __builtin_bswap32(value), __builtin_bswap32(value));

        Initialize(cols, rows, count);

        int counter = 0;
        while (!feof(fimg) && !feof(flabel) && counter < m_count) {
            float* imageBuffer = &m_imageBuffer[counter * m_imageSize];

            for (int j = 0; j < m_height; ++j) {
                for (int i = 0; i < m_width; ++i) {
                    uint8_t pixel;
                    if (fread(&pixel, sizeof(uint8_t), 1, fimg) != 1) {
                        printf("Failed to read pixel (%d, %d) from image %d in %s",
                            i, j, counter, imageFile);
                        return 1;
                    }
                    imageBuffer[j * m_width + i] = pixel;
                }
            }

            uint8_t cat;
            fread(&cat, sizeof(uint8_t), 1, flabel);
            m_categoryBuffer[counter] = cat;

            ++counter;
        }

        return 0;
    }

private:
    void Initialize(const int width, const int height, const int count) {
        m_width = width;
        m_height = height;
        m_imageSize = m_width * m_height;
        m_count = count;

        m_buffer = (float*)malloc(m_count * m_imageSize * sizeof(float));
        m_imageBuffer = m_buffer;
        m_categoryBuffer = (uint8_t*)malloc(m_count * sizeof(uint8_t));
    }

    int m_count;
    int m_width;
    int m_height;
    int m_imageSize;
    float* m_buffer;
    float* m_imageBuffer;
    uint8_t* m_categoryBuffer;

    static const int c_categoryCount = 10;
};

void LoadMNISTData(const std::string& image_fname, const std::string& label_fname,
    int& n_images, int& c, int& h, int& w, int& n_classes,
    float** image_data, float** label_data, bool verbose = true) {
    MNISTDataset data;
    assert(data.Parse(image_fname.c_str(), label_fname.c_str(), verbose) == 0);

    n_images = data.GetImageCount();
    c = 1;
    h = data.GetImageHeight();
    w = data.GetImageWidth();
    n_classes = 10;

    int X_size = n_images * c * h * w;
    *image_data = new float[X_size];
    for (int i = 0; i < X_size; ++i)
        (*image_data)[i] = data.GetImageData()[i] / 255.0f;

    int Y_size = n_images * n_classes;
    *label_data = new float[Y_size];
    std::fill(*label_data, *label_data + Y_size, 0.0f);
    for (int i = 0; i < n_images; ++i) {
        int y = data.GetCategoryData()[i];
        (*label_data)[i * n_classes + y] = 1.0f;
    }
}
