const { defineConfig } = require('@vue/cli-service')

const path = require("path");
function resolve(dir) {
  return path.join(__dirname, dir);
}
module.exports = defineConfig({
    devServer: {
      proxy: {
        '/api': {
          target: 'http://localhost:3000', // 后端服务的地址
          changeOrigin: true,
          pathRewrite: { '^/api': '' } // 将 /api 前缀重写为空字符串
        }
      }
    },
    configureWebpack: {
      name:'hospital', // 你的应用名称
      resolve: {
        alias: {
          '@': resolve('src'),
          '@UI': resolve('UI')
        }
      },
      module: {
        rules: [
          {
            test: /\.ts$/,
            loader: 'ts-loader',
            options: {
              appendTsSuffixTo: [/\.vue$/],
            },
            exclude: /node_modules/,
          },
        ],
      },
    },
    chainWebpack: config => {
      config.when(process.env.NODE_ENV !== 'development', config => {
        config.optimization.splitChunks({
          chunks: 'all',
          cacheGroups: {
            libs: {
              name: 'chunk-libs',
              test: /[\\/]node_modules[\\/]/,
              priority: 10,
              chunks: 'initial'
            },
            commons: {
              name: 'chunk-commons',
              test: resolve('src/components'),
              minChunks: 3,
              priority: 5,
              reuseExistingChunk: true
            }
          }
        });
        config.optimization.runtimeChunk('single');
      });
    }
  });