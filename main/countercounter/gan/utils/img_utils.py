from torchvision.utils import save_image


def save_images(epoch, originals, generated, generated_rescaled, dir, sample_size):
    # incoming: list of tensors (might be [(n tensors), (n tensors), ...]
    third_images = len(generated_rescaled) > 0

    count = 0

    for idx in range(len(originals)):
        if count > sample_size:
            break

        orig = originals[idx]
        gen_eval = generated[idx]

        if third_images:
            gen_rescaled = generated_rescaled[idx]

        for inner_idx in range(orig.size(0)):
            if count > sample_size:
                break

            o = orig[inner_idx]
            gnd = gen_eval[inner_idx]

            save_image(o, dir + f"/{epoch}_nr_{count}_original.png")
            # save_image(gnd, dir + f"/{epoch}_nr_{count}_generated.png")

            if third_images:
                res = gen_rescaled[inner_idx]
                save_image(res, dir + f'/{epoch}_nr_{count}_generated_rescaled.png')

            count += 1