Shader "Unlit/KuwaharaFilter"
{
    Properties
    {
        _MainTex ("Texture", 2D) = "white" {}
        _Size ("Filter Size", Range(1, 25)) = 5
    }

    SubShader
    {
        Cull Off
        ZWrite On
        ZTest Always

        Tags { "RenderType" = "Opaque" "Renderpipeline" = "UniversalPipeline" }
    
        Pass
        {
            HLSLPROGRAM
            #pragma vertex vert
            #pragma fragment frag

            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"
            #include "Packages/com.unity.render-pipelines.core/Runtime/Utilities/Blit.hlsl"

            #if SHADER_API_GLES
                struct FilterAttributes
                {
                    float4 vertex : POSITION;
                    float2 uv : TEXCOORD0;
                };
            #else
                struct FilterAttributes
                {
                    uint vertex : SV_VertexID;
                };
            #endif
        
            struct FilterVaryings
            {
                float2 uv : TEXCOORD0;
                float4 vertex : SV_POSITION;
            };

            float _Size;
            float4 _BlitTexture_TexelSize;

            FilterVaryings vert(FilterAttributes v)
            {
                FilterVaryings OUT;
                #if SHADER_API_GLES
                    float4 pos = input.vertex;
                    float2 uv = input.uv;
                #else
                    float4 pos = GetFullScreenTriangleVertexPosition(v.vertex);
                    float2 uv = GetFullScreenTriangleTexCoord(v.vertex);
                #endif

                OUT.vertex = pos;
                OUT.uv = uv;
                return OUT;
            }

            float4 frag(FilterVaryings i) : SV_TARGET
            {
                // uvのテクセルの座標を取得
                float uTexel = _BlitTexture_TexelSize.x;
                float vTexel = _BlitTexture_TexelSize.y;
            
                // 4エリアの正方形の1辺の長さを求める
                // 点Aから右上、右下、左上、左下の4区画を調べる処理を行うためにその区画の大きさを決める
                int areaSize = floor(_Size * 0.5);

                // 長さが0なら何もしない
                if(areaSize == 0)
                {
                    return SAMPLE_TEXTURE2D(_BlitTexture, sampler_LinearRepeat, i.uv);
                }

                // 各エリアの全ピクセルのrgb値から平均と分散を求める
                // a : 左上
                float3 aavg = float3(0,0,0);
                float3 avar = float3(0,0,0);
                for (int av = 1; av <= areaSize; av++) 
                {
                    for (int au = 1; au <= areaSize; au++) 
                    {
                        float3 pick = SAMPLE_TEXTURE2D(_BlitTexture, sampler_LinearRepeat, i.uv + float2(-au*uTexel, av*vTexel));
                        aavg += pick;
                        avar += pick * pick;
                    }
                }
                aavg /= areaSize * areaSize; // ここは平均を求めていそう
                avar = avar / (areaSize * areaSize) - aavg * aavg; // ここで分散を求めている

                // b : 右上
                float3 bavg = float3(0,0,0);
                float3 bvar = float3(0,0,0);
                for (int bv = 1; bv <= areaSize; bv++) 
                {
                    for (int bu = 1; bu <= areaSize; bu++) 
                    {
                        float3 pick = SAMPLE_TEXTURE2D(_BlitTexture, sampler_LinearRepeat, i.uv + float2(bu*uTexel, bv*vTexel));
                        bavg += pick;
                        bvar += pick * pick;
                    }
                }
                bavg /= areaSize * areaSize;
                bvar = bvar / (areaSize * areaSize) - bavg * bavg;

                // c : 左下
                float3 cavg = float3(0,0,0);
                float3 cvar = float3(0,0,0);
                for (int cv = 1; cv <= areaSize; cv++) 
                {
                    for (int cu = 1; cu <= areaSize; cu++) 
                    {
                        float3 pick = SAMPLE_TEXTURE2D(_BlitTexture, sampler_LinearRepeat, i.uv + float2(-cu*uTexel, -cv*vTexel));
                        cavg += pick;
                        cvar += pick * pick;
                    }
                }
                cavg /= areaSize * areaSize;
                cvar = cvar / (areaSize * areaSize) - cavg * cavg;

                // d : 右下
                float3 davg = float3(0,0,0);
                float3 dvar = float3(0,0,0);
                for (int dv = 1; dv <= areaSize; dv++) 
                {
                    for (int du = 1; du <= areaSize; du++) 
                    {
                        float3 pick = SAMPLE_TEXTURE2D(_BlitTexture, sampler_LinearRepeat, i.uv + float2(du*uTexel, -dv*vTexel));
                        davg += pick;
                        dvar += pick * pick;
                    }
                }
                davg /= areaSize * areaSize;
                dvar = dvar / (areaSize * areaSize) - davg * davg;

                // 各rgb値について、4エリアで最も分散が小さいエリアの平均値を取得
                // r
                float r = lerp(aavg.r, bavg.r, step(bvar.r, avar.r));
                r = lerp(r, cavg.r, step(cvar.r, min(avar.r, bvar.r)));
                r = lerp(r, davg.r, step(dvar.r, min(cvar.r, min(avar.r, bvar.r))));

                // g
                float g = lerp(aavg.g, bavg.g, step(bvar.g, avar.g));
                g = lerp(g, cavg.g, step(cvar.g, min(avar.g, bvar.g)));
                g = lerp(g, davg.g, step(dvar.g, min(cvar.g, min(avar.g, bvar.g))));

                // b
                float b = lerp(aavg.b, bavg.b, step(bvar.b, avar.b));
                b = lerp(b, cavg.b, step(cvar.b, min(avar.b, bvar.b)));
                b = lerp(b, davg.b, step(dvar.b, min(cvar.b, min(avar.b, bvar.b))));

                // 取得したrgb値を最終的な色としてて出力
                float4 col = float4(r,g,b,1);
                return col;
            }
            ENDHLSL
        }
    }
}