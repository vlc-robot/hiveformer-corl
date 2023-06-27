// next.config.js
const isProd = process.env.NODE_ENV === 'production'

module.exports = {
  assetPrefix: isProd ? '/hiveformer-corl/' : '',
  images: {
    unoptimized: true,
  },
}
