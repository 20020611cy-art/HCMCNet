fused_feats = []

for pre, post in zip(pre_feats, post_feats):

    diff = torch.abs(pre - post)
    attn = torch.sigmoid(diff)

    fused = torch.cat([
        attn * pre,
        attn * post,
        diff,
    ], dim=1)

    fused_feats.append(fused)

out = self.decode_head(fused_feats)



fused_feats = []

for pre, post in zip(pre_feats, post_feats):

    mul = pre * post
    diff = torch.abs(pre - post)
    attn = torch.sigmoid(diff)

    fused = torch.cat([
        attn * pre,
        attn * post,
        mul,
    ], dim=1)

    fused_feats.append(fused)

out = self.decode_head(fused_feats)


fused_feats = []
            for i, (pre, post) in enumerate(zip(pre_feats, post_feats)):

                diff = torch.abs(pre - post)
                attn = torch.sigmoid(diff)
                mul = pre * post

                if i < 2:  # 浅层
                    fused = torch.cat([
                        attn * pre,
                        attn * post,
                        diff,
                    ], dim=1)
                else:  # 深层
                    fused = torch.cat([
                        attn * pre,
                        attn * post,
                        mul,
                    ], dim=1)

                fused_feats.append(fused)