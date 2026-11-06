"""CPU tests for the zero-shot knowledge-consistency loss and its teacher-confidence reweighting."""
import types
import unittest
import torch
from torch import nn
from torch.nn import functional as F
from trainers.maple import MaPLe
from tests.test_gating import config

torch.set_num_threads(1)


class StubTeacher(nn.Module):
    dtype = torch.float32

    def __init__(self, features):
        super().__init__()
        self.features = features
        self.logit_scale = torch.tensor(2.0).log()  # CLIP stores log(scale); scale = 2

    def encode_image(self, image):
        return self.features


def fake_trainer(reweight):
    cfg = config()
    m = cfg.TRAINER.MAPLE
    m.KC_TEXT_WEIGHT, m.KC_IMAGE_WEIGHT, m.KC_LOGIT_WEIGHT, m.KC_REWEIGHT = 25.0, 10.0, 1.0, reweight
    torch.manual_seed(0)
    zs_image = F.normalize(torch.randn(4, 8), dim=-1)
    zs_text = F.normalize(torch.randn(3, 8), dim=-1)
    return types.SimpleNamespace(cfg=cfg, _zs_teacher=StubTeacher(zs_image), _zs_text=zs_text), zs_image, zs_text


def inputs():
    torch.manual_seed(1)
    image_features = F.normalize(torch.randn(4, 8), dim=-1)
    text_features = F.normalize(torch.randn(3, 8), dim=-1)
    logits = 5 * image_features @ text_features.t()
    return torch.zeros(4, 3, 2, 2), torch.tensor([0, 1, 2, 1]), logits, image_features, text_features


class KnowledgeConsistencyTests(unittest.TestCase):
    def test_unweighted_matches_original_formula(self):
        trainer, zs_image, zs_text = fake_trainer('none')
        image, label, logits, f_v, f_t = inputs()
        parts = MaPLe._consistency_loss(trainer, image, label, logits, f_v, f_t)
        zs_logits = 2.0 * zs_image @ zs_text.t()
        torch.testing.assert_close(parts['kc_text'], 25 * F.l1_loss(f_t, zs_text))
        torch.testing.assert_close(parts['kc_image'], 10 * F.l1_loss(f_v, zs_image))
        torch.testing.assert_close(parts['kc_logit'], F.kl_div(
            F.log_softmax(logits, 1), F.log_softmax(zs_logits, 1), reduction='sum', log_target=True) / logits.numel())
        self.assertNotIn('kc_w', parts)

    def test_teacher_confidence_weights(self):
        trainer, zs_image, zs_text = fake_trainer('teacher_conf')
        image, label, logits, f_v, f_t = inputs()
        parts = MaPLe._consistency_loss(trainer, image, label, logits, f_v, f_t)
        zs_logits = 2.0 * zs_image @ zs_text.t()
        w = F.softmax(zs_logits, 1).gather(1, label[:, None]).squeeze(1)
        torch.testing.assert_close(parts['kc_w'], w.mean())
        torch.testing.assert_close(parts['kc_image'], 10 * (w * (f_v - zs_image).abs().mean(1)).mean())
        torch.testing.assert_close(parts['kc_text'], 25 * w.mean() * F.l1_loss(f_t, zs_text))
        # The text scale is the running mean of w over every sample seen so far.
        label2 = torch.tensor([2, 2, 0, 0])
        second = MaPLe._consistency_loss(trainer, image, label2, logits, f_v, f_t)
        w2 = F.softmax(zs_logits, 1).gather(1, label2[:, None]).squeeze(1)
        self.assertEqual(trainer._kc_w_count, 8)
        torch.testing.assert_close(second['kc_text'], 25 * torch.cat([w, w2]).mean() * F.l1_loss(f_t, zs_text))

    def test_invalid_reweight_rejected(self):
        trainer, _, _ = fake_trainer('bogus')
        with self.assertRaisesRegex(ValueError, 'KC_REWEIGHT'):
            MaPLe._consistency_loss(trainer, *inputs())


if __name__ == '__main__':
    unittest.main()
